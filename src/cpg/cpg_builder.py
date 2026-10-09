"""
Stage 1: Code Property Graph (CPG) Construction
Constructs a unified representation combining syntax, control flow, and dependencies.
"""

import os
import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
import networkx as nx
from collections import Counter, defaultdict
from loguru import logger

import javalang

from src.cpg.call_resolver import (
    CHAINED, PLAIN, QUALIFIED, SUPER, THIS,
    CallResolver, PendingCall, coverage, simple_type_name,
)
from src.refactor.extent import member_extent


@dataclass
class CodeNode:
    """Represents a node in the CPG"""
    node_id: str
    node_type: str  # "method", "class", "variable", "statement", etc.
    name: str
    file_path: str
    line_number: int
    code_snippet: str  # exactly ONE line - see _extract_snippet
    # Full declaration source for class/method nodes (modifiers and annotations
    # through the closing brace). Kept SEPARATE from code_snippet on purpose:
    # smell_detector's duplicate-code Jaccard and boolean-operator thresholds are
    # calibrated on single-line statement snippets, so widening code_snippet would
    # silently change detector behaviour. Empty for nodes with no meaningful extent.
    source_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeEdge:
    """Represents an edge in the CPG"""
    source_id: str
    target_id: str
    edge_type: str  # "calls", "defines", "uses", "inherits", etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


class CodePropertyGraph:
    """
    Constructs comprehensive Code Property Graph (CPG) that unifies:
    - Syntax information (AST)
    - Control flow (CFG)
    - Data flow (DFG)
    - Call graphs
    - Inter-file dependencies
    """
    
    def __init__(self, language: str = "java", include_data_flow: bool = True,
                 include_control_flow: bool = True, include_call_graph: bool = True):
        self.language = language
        self.include_data_flow = include_data_flow
        self.include_control_flow = include_control_flow
        self.include_call_graph = include_call_graph
        
        self.graph = nx.MultiDiGraph()
        self.nodes: Dict[str, CodeNode] = {}
        self.edges: List[CodeEdge] = []
        self.file_map: Dict[str, List[str]] = defaultdict(list)  # file -> node_ids

        # Call-graph resolution (see src/cpg/call_resolver.py). Invocations are
        # COLLECTED during the first pass and resolved during the second,
        # because an invocation's target is frequently declared in a file that
        # has not been parsed yet - resolving eagerly is what previously forced
        # the builder to assume every call targeted the caller's own file.
        self.call_resolver = CallResolver()
        self._pending_calls: List[PendingCall] = []
        self._pending_inherits: List[Tuple[str, str]] = []  # (class node id, supertype name)
        self._class_nodes_by_name: Dict[str, List[str]] = defaultdict(list)
        # Invocations deliberately left unresolved, each with its reason, and
        # the tally of which rule decided each call. Reported by
        # get_statistics() so the call graph's real coverage is visible.
        self.unresolved_calls: List[Dict[str, Any]] = []
        self.call_resolution_stats: Counter = Counter()

        logger.info(f"Initialized CPG for language: {language}")
    
    def build_from_directory(self, directory_path: str) -> None:
        """
        Build CPG from all source files in a directory
        """
        logger.info(f"Building CPG from directory: {directory_path}")
        
        file_extension = f".{self.language}"
        source_files = Path(directory_path).rglob(f"*{file_extension}")
        
        # First pass: Extract nodes from all files
        for file_path in source_files:
            try:
                self._process_file(str(file_path))
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
        
        # Second pass: Build inter-file dependencies
        self._build_cross_file_dependencies()
        
        logger.info(f"CPG construction complete. Nodes: {len(self.nodes)}, Edges: {len(self.edges)}")
    
    def _process_file(self, file_path: str) -> None:
        """Process a single source file"""
        logger.debug(f"Processing file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if self.language == "java":
            self._process_java_file(file_path, content)
        elif self.language == "python":
            self._process_python_file(file_path, content)
        else:
            logger.warning(f"Language {self.language} not yet supported")
    
    @staticmethod
    def _supertype_names(node: Any) -> List[str]:
        """Simple names of a type declaration's supertypes (extends + implements).

        `extends` is a single reference on a class but a LIST on an interface,
        so both shapes are accepted rather than assumed.
        """
        names: List[str] = []
        extends = getattr(node, "extends", None)
        if extends is not None:
            for parent in (extends if isinstance(extends, list) else [extends]):
                name = simple_type_name(parent)
                if name:
                    names.append(name)
        for implemented in (getattr(node, "implements", None) or []):
            name = simple_type_name(implemented)
            if name:
                names.append(name)
        return names

    def _index_type_declarations(self, file_path: str, tree: Any) -> None:
        """Register every type declared in this file with the call resolver.

        Interfaces and enums are indexed for RESOLUTION but are not added as
        CPG nodes - only ClassDeclaration becomes a "class" node, which is the
        pre-existing node-type contract that smell detection, the labeled
        dataset and the GNN's node-type vocabulary all depend on. Their method
        declarations are already captured as method nodes (the builder filters
        MethodDeclaration over the whole tree regardless of the enclosing
        type), so indexing the type name alone is enough to resolve
        `service.save()` where `service` is declared as an interface - which in
        these Spring-style repos is the common case.
        """
        for declaration_type in (javalang.tree.ClassDeclaration,
                                 javalang.tree.InterfaceDeclaration,
                                 javalang.tree.EnumDeclaration):
            for _path, node in tree.filter(declaration_type):
                self.call_resolver.index_class(node.name, file_path,
                                               self._supertype_names(node))
                # Type-level annotations, so a call to a member Lombok
                # SYNTHESISES (a @Data accessor, @Slf4j's `log`) can be
                # attributed to code generation instead of being reported as a
                # resolution failure - the member genuinely is not in the source.
                self.call_resolver.index_type_annotations(file_path, {
                    annotation.name
                    for annotation in (getattr(node, "annotations", None) or [])
                    if getattr(annotation, "name", None)
                })

    def _process_java_file(self, file_path: str, content: str) -> None:
        """Extract entities and relationships from Java file"""
        try:
            tree = javalang.parse.parse(content)
            # Index every declared type FIRST: the field/local type lookups and
            # supertype walks used by call resolution are all keyed off this.
            self._index_type_declarations(file_path, tree)
            
            # Extract class definitions
            for path, node in tree.filter(javalang.tree.ClassDeclaration):
                class_node_id = f"class_{file_path}_{node.name}"
                source_text, extent = self._extract_source_text(content, node)
                self._add_node(
                    node_id=class_node_id,
                    node_type="class",
                    name=node.name,
                    file_path=file_path,
                    line_number=node.position[0] if node.position else 0,
                    code_snippet=self._extract_snippet(content, node.position),
                    source_text=source_text,
                    extent=extent
                )
                self._class_nodes_by_name[node.name].append(class_node_id)
                for supertype in self._supertype_names(node):
                    self._pending_inherits.append((class_node_id, supertype))

            # Extract method definitions
            for path, node in tree.filter(javalang.tree.MethodDeclaration):
                method_node_id = f"method_{file_path}_{node.name}"
                source_text, extent = self._extract_source_text(content, node)
                self._add_node(
                    node_id=method_node_id,
                    node_type="method",
                    name=node.name,
                    file_path=file_path,
                    line_number=node.position[0] if node.position else 0,
                    code_snippet=self._extract_snippet(content, node.position),
                    source_text=source_text,
                    extent=extent
                )
                self.call_resolver.index_method(method_node_id, node.name, file_path)
                # Build control-flow, data-flow and call edges for this method body
                self._process_method_body(file_path, content, method_node_id, node)
            
            # Extract field definitions
            for path, node in tree.filter(javalang.tree.FieldDeclaration):
                field_type = simple_type_name(getattr(node, "type", None))
                for decl in node.declarators:
                    field_node_id = f"field_{file_path}_{decl.name}"
                    self._add_node(
                        node_id=field_node_id,
                        node_type="field",
                        name=decl.name,
                        file_path=file_path,
                        line_number=node.position[0] if node.position else 0,
                        code_snippet=self._extract_snippet(content, node.position)
                    )
                    self.call_resolver.index_field_type(file_path, decl.name, field_type)
        
        except Exception as e:
            logger.error(f"Error parsing Java file {file_path}: {e}")
    
    def _process_python_file(self, file_path: str, content: str) -> None:
        """Extract entities and relationships from Python file"""
        try:
            tree = ast.parse(content)
            
            # Extract class definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_node_id = f"class_{file_path}_{node.name}"
                    self._add_node(
                        node_id=class_node_id,
                        node_type="class",
                        name=node.name,
                        file_path=file_path,
                        line_number=node.lineno,
                        code_snippet=""
                    )
                
                elif isinstance(node, ast.FunctionDef):
                    func_node_id = f"function_{file_path}_{node.name}"
                    self._add_node(
                        node_id=func_node_id,
                        node_type="function",
                        name=node.name,
                        file_path=file_path,
                        line_number=node.lineno,
                        code_snippet=""
                    )
        
        except Exception as e:
            logger.error(f"Error parsing Python file {file_path}: {e}")
    
    def _process_method_body(self, file_path: str, content: str,
                              method_node_id: str, method_node: Any) -> None:
        """Build control-flow, data-flow and call-graph edges for a single method"""
        var_decls: Dict[str, str] = {}
        
        # Parameters count as initial variable definitions flowing from the method
        if self.include_data_flow:
            for param in (method_node.parameters or []):
                param_node_id = f"variable_{file_path}_{method_node.name}_{param.name}"
                self._add_node(
                    node_id=param_node_id,
                    node_type="variable",
                    name=param.name,
                    file_path=file_path,
                    line_number=method_node.position[0] if method_node.position else 0,
                    code_snippet=""
                )
                var_decls[param.name] = param_node_id
                self.edges.append(CodeEdge(
                    source_id=method_node_id,
                    target_id=param_node_id,
                    edge_type="data_flow",
                    metadata={"role": "parameter"}
                ))
        
        # Walk the method body to build sequential/branching control flow and def-use data flow
        if method_node.body and (self.include_control_flow or self.include_data_flow):
            counter = [0]
            self._process_statements(
                file_path, content, method_node.name, method_node.body,
                method_node_id, var_decls, counter, first_edge_kind="entry"
            )
        
        # Declared types of parameters and locals, needed to resolve `x.m()` to
        # the class x was declared as. Collected independently of the data-flow
        # walk above so call resolution does not silently degrade when
        # include_data_flow is off.
        if self.include_call_graph:
            self._index_local_types(file_path, method_node)

        # Call graph: every invocation inside this method is attributed to it,
        # but only COLLECTED here. Targets are resolved in the second pass by
        # src/cpg/call_resolver.py, once every file's declarations are indexed -
        # this used to synthesise `method_<this file>_<name>`, which assumed
        # every call stayed inside the calling file and so produced zero
        # cross-file call edges plus thousands of dangling targets.
        if self.include_call_graph:
            try:
                self._collect_invocations(file_path, method_node_id, method_node)
            except Exception as e:
                logger.debug(f"Error extracting method calls for {method_node_id}: {e}")

    def _index_local_types(self, file_path: str, method_node: Any) -> None:
        """Record the declared type of every named binding inside a method.

        Four javalang shapes introduce a typed name, and missing any of them
        makes `x.m()` unresolvable for that x:
          * FormalParameter          - `void m(Batch b)`
          * LocalVariableDeclaration - `Batch b = ...;`
          * VariableDeclaration      - `for (Batch b : batches)` and `for (int i...)`
          * TryResource              - `try (Resource r = open())`
          * CatchClauseParameter     - `catch (MyError e)`, whose `.types` is a
            list of plain STRINGS, not type nodes, so it needs its own handling.

        Lambda parameters are deliberately absent: javalang exposes an inferred
        lambda parameter as a MemberReference with `name=None`, so neither the
        name nor the type is recoverable, and a lambda receiver's type needs
        real inference anyway. Those calls stay unresolved by design.
        """
        method_name = method_node.name

        def record(name: Optional[str], type_name: Optional[str]) -> None:
            if name:
                self.call_resolver.index_local_type(file_path, method_name, name, type_name)

        for param in (method_node.parameters or []):
            record(param.name, simple_type_name(getattr(param, "type", None)))

        try:
            for declaration_type in (javalang.tree.LocalVariableDeclaration,
                                     javalang.tree.VariableDeclaration,
                                     javalang.tree.TryResource):
                for _path, declaration in method_node.filter(declaration_type):
                    type_name = simple_type_name(getattr(declaration, "type", None))
                    declarators = getattr(declaration, "declarators", None)
                    if declarators:
                        for declarator in declarators:
                            record(declarator.name, type_name)
                    else:  # TryResource carries its name directly
                        record(getattr(declaration, "name", None), type_name)

            for _path, catch_param in method_node.filter(javalang.tree.CatchClauseParameter):
                types = getattr(catch_param, "types", None) or []
                # A multi-catch (`catch (A | B e)`) has no single type, so only a
                # single-type clause is specific enough to index.
                if len(types) == 1:
                    record(catch_param.name, str(types[0]).split(".")[-1])

            # Lambda parameters: javalang models an inferred parameter as a
            # MemberReference, so the NAME is on `.member` (not `.name`) and the
            # type is absent. Recording the name lets an unresolvable receiver
            # be attributed to missing type inference instead of looking like a
            # binding the indexer overlooked.
            for _path, lambda_expr in method_node.filter(javalang.tree.LambdaExpression):
                for parameter in (lambda_expr.parameters or []):
                    declared = simple_type_name(getattr(parameter, "type", None))
                    name = getattr(parameter, "name", None) or getattr(parameter, "member", None)
                    if not name:
                        continue
                    if declared:
                        record(name, declared)
                    else:
                        self.call_resolver.index_untyped_name(file_path, method_name, name)
        except Exception as e:
            logger.debug(f"Error indexing local types for {file_path}:{method_name}: {e}")

    @staticmethod
    def _invocation_kind(path: Any, invocation: Any) -> str:
        """Classify an invocation's receiver from javalang's AST shape.

        javalang encodes the receiver three different ways, and conflating them
        is what makes naive resolution unsound:
          * qualifier == ""   -> an unqualified call, `m()`
          * qualifier == name -> an explicit receiver, `x.m()` or `Foo.m()`
          * qualifier is None -> this node is a SELECTOR of something else, which
            is either `this.m()` (enclosing node is `This`) or a chained call
            such as `getX().m()` (enclosing node is another invocation). Only
            the former is resolvable as an unqualified call.
        """
        qualifier = invocation.qualifier
        if qualifier:
            return QUALIFIED
        if qualifier == "":
            return PLAIN
        enclosing = [node for node in path if not isinstance(node, (list, tuple))]
        if enclosing and isinstance(enclosing[-1], javalang.tree.This):
            return THIS
        return CHAINED

    def _collect_invocations(self, file_path: str, method_node_id: str,
                             method_node: Any) -> None:
        """Queue every invocation in this method for second-pass resolution."""
        for path, invocation in method_node.filter(javalang.tree.MethodInvocation):
            self._pending_calls.append(PendingCall(
                caller_node_id=method_node_id,
                caller_file=file_path,
                caller_method=method_node.name,
                member=invocation.member,
                kind=self._invocation_kind(path, invocation),
                qualifier=invocation.qualifier,
                line=invocation.position[0] if invocation.position else 0,
            ))
        # `super.m()` is a DIFFERENT javalang node type and was previously
        # missed entirely, so inherited calls never appeared in the call graph.
        for _path, invocation in method_node.filter(javalang.tree.SuperMethodInvocation):
            self._pending_calls.append(PendingCall(
                caller_node_id=method_node_id,
                caller_file=file_path,
                caller_method=method_node.name,
                member=invocation.member,
                kind=SUPER,
                qualifier=None,
                line=invocation.position[0] if invocation.position else 0,
            ))
    
    def _normalize_statements(self, statements: Any) -> List[Any]:
        """Flatten a statement, block, or list of statements into an ordered list of atomic statements"""
        if statements is None:
            return []
        if isinstance(statements, list):
            flattened: List[Any] = []
            for stmt in statements:
                flattened.extend(self._normalize_statements(stmt))
            return flattened
        if isinstance(statements, javalang.tree.BlockStatement):
            return self._normalize_statements(statements.statements)
        return [statements]
    
    def _process_statements(self, file_path: str, content: str, method_name: str,
                             statements: Any, prev_id: str, var_decls: Dict[str, str],
                             counter: List[int], first_edge_kind: str = "sequence") -> str:
        """
        Wire control-flow (sequential + branching) and data-flow (def-use) edges for a
        sequence of statements. Returns the id of the last statement processed so callers
        can keep chaining sequential flow.
        """
        edge_kind = first_edge_kind
        for stmt in self._normalize_statements(statements):
            counter[0] += 1
            stmt_id = f"stmt_{file_path}_{method_name}_{counter[0]}"
            position = getattr(stmt, "position", None)
            line = position[0] if position else 0
            self._add_node(
                node_id=stmt_id,
                node_type="statement",
                name=type(stmt).__name__,
                file_path=file_path,
                line_number=line,
                code_snippet=self._extract_snippet(content, position) if position else ""
            )
            
            if self.include_control_flow:
                self.edges.append(CodeEdge(
                    source_id=prev_id,
                    target_id=stmt_id,
                    edge_type="control_flow",
                    metadata={"kind": edge_kind}
                ))
            edge_kind = "sequence"  # subsequent siblings simply follow on
            
            if self.include_data_flow:
                self._process_data_flow_for_statement(file_path, stmt, stmt_id, var_decls)
            
            prev_id = stmt_id
            
            # Recurse into branch/loop/try bodies as sub-chains hanging off this statement
            self._process_nested_branches(file_path, content, method_name, stmt, stmt_id, var_decls, counter)
        
        return prev_id
    
    def _process_nested_branches(self, file_path: str, content: str, method_name: str,
                                  stmt: Any, stmt_id: str, var_decls: Dict[str, str],
                                  counter: List[int]) -> None:
        """Recurse into the bodies of branching/looping/exception-handling statements"""
        if not (self.include_control_flow or self.include_data_flow):
            return
        
        if isinstance(stmt, javalang.tree.IfStatement):
            if stmt.then_statement:
                self._process_statements(file_path, content, method_name, stmt.then_statement,
                                          stmt_id, var_decls, counter, first_edge_kind="branch_true")
            if stmt.else_statement:
                self._process_statements(file_path, content, method_name, stmt.else_statement,
                                          stmt_id, var_decls, counter, first_edge_kind="branch_false")
        
        elif isinstance(stmt, (javalang.tree.WhileStatement, javalang.tree.DoStatement, javalang.tree.ForStatement)):
            if stmt.body:
                self._process_statements(file_path, content, method_name, stmt.body,
                                          stmt_id, var_decls, counter, first_edge_kind="loop_body")
        
        elif isinstance(stmt, javalang.tree.TryStatement):
            if stmt.block:
                self._process_statements(file_path, content, method_name, stmt.block,
                                          stmt_id, var_decls, counter, first_edge_kind="try_body")
            for catch in (stmt.catches or []):
                if catch.block:
                    self._process_statements(file_path, content, method_name, catch.block,
                                              stmt_id, var_decls, counter, first_edge_kind="catch_body")
            if stmt.finally_block:
                self._process_statements(file_path, content, method_name, stmt.finally_block,
                                          stmt_id, var_decls, counter, first_edge_kind="finally_body")
        
        elif isinstance(stmt, javalang.tree.SwitchStatement):
            for case in (stmt.cases or []):
                case_statements = getattr(case, "statements", None) or []
                if case_statements:
                    self._process_statements(file_path, content, method_name, case_statements,
                                              stmt_id, var_decls, counter, first_edge_kind="case_body")
    
    def _process_data_flow_for_statement(self, file_path: str, stmt: Any, stmt_id: str,
                                          var_decls: Dict[str, str]) -> None:
        """Create definition edges for declared variables and use edges for referenced variables"""
        if isinstance(stmt, javalang.tree.LocalVariableDeclaration):
            position = getattr(stmt, "position", None)
            line = position[0] if position else 0
            for decl in stmt.declarators:
                var_node_id = f"variable_{file_path}_{decl.name}_{stmt_id}"
                self._add_node(
                    node_id=var_node_id,
                    node_type="variable",
                    name=decl.name,
                    file_path=file_path,
                    line_number=line,
                    code_snippet=""
                )
                var_decls[decl.name] = var_node_id
                self.edges.append(CodeEdge(
                    source_id=stmt_id,
                    target_id=var_node_id,
                    edge_type="data_flow",
                    metadata={"role": "definition"}
                ))
        
        try:
            for _, ref in stmt.filter(javalang.tree.MemberReference):
                var_name = ref.member
                if var_name in var_decls:
                    self.edges.append(CodeEdge(
                        source_id=var_decls[var_name],
                        target_id=stmt_id,
                        edge_type="data_flow",
                        metadata={"role": "use"}
                    ))
        except Exception as e:
            logger.debug(f"Error extracting data flow references for {stmt_id}: {e}")
    
    def _add_node(self, node_id: str, node_type: str, name: str,
                  file_path: str, line_number: int, code_snippet: str,
                  source_text: str = "", extent: Optional[Dict[str, Any]] = None) -> None:
        """Add a node to the CPG.

        `source_text` and `extent` are keyword-with-default so existing
        positional callers (including tests) keep working unchanged.
        """
        node = CodeNode(
            node_id=node_id,
            node_type=node_type,
            name=name,
            file_path=file_path,
            line_number=line_number,
            code_snippet=code_snippet,
            source_text=source_text,
            metadata={"extent": extent} if extent else {}
        )
        
        self.nodes[node_id] = node
        self.file_map[file_path].append(node_id)
        
        # Add to NetworkX graph
        self.graph.add_node(node_id, **{
            "type": node_type,
            "name": name,
            "file": file_path,
            "line": line_number
        })
    
    def _build_cross_file_dependencies(self) -> None:
        """Second pass: resolve queued invocations and inheritance, then
        materialise every edge into the NetworkX graph.

        Runs only once the first pass has indexed every declaration in the
        corpus, which is what makes genuine cross-file resolution possible.
        """
        logger.info("Building cross-file dependencies...")

        self._resolve_pending_calls()
        self._resolve_pending_inherits()

        # Only edges whose BOTH endpoints are real nodes are added.
        # `nx.add_edge` silently creates any node it has not seen, so an
        # unresolvable target used to materialise a phantom node - 434 of them
        # on one 66-file repo - leaving graph.number_of_nodes() permanently
        # larger than len(self.nodes) and every reported node count wrong.
        skipped = 0
        for edge in self.edges:
            if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
                skipped += 1
                continue
            self.graph.add_edge(edge.source_id, edge.target_id,
                                type=edge.edge_type, **edge.metadata)
        if skipped:
            logger.warning(f"{skipped} edge(s) referenced a node outside the graph and "
                           f"were not materialised (this should be 0 - investigate)")

    def _resolve_pending_calls(self) -> None:
        """Turn queued invocations into `calls` edges via the global index."""
        if not self._pending_calls:
            return

        resolved, unresolved, stats = self.call_resolver.resolve_all(self._pending_calls)
        self.unresolved_calls = unresolved
        self.call_resolution_stats = stats

        cross_file = 0
        for call, resolution in resolved:
            target_file = self.nodes[resolution.target_node_id].file_path
            is_cross_file = target_file != call.caller_file
            cross_file += int(is_cross_file)
            self.edges.append(CodeEdge(
                source_id=call.caller_node_id,
                target_id=resolution.target_node_id,
                edge_type="calls",
                metadata={
                    "qualifier": call.qualifier or "",
                    "receiver_kind": call.kind,
                    # Which rule produced this edge, so a weaker inference
                    # (unique_global) can be told apart from a strong one
                    # (same_file, qualifier_type) downstream.
                    "resolution": resolution.strategy,
                    "cross_file": is_cross_file,
                },
            ))

        total = len(self._pending_calls)
        logger.info(
            f"Call resolution: {len(resolved)}/{total} invocations resolved "
            f"({cross_file} cross-file), {len(unresolved)} left unresolved")
        logger.debug(f"Call resolution strategies: {dict(stats)}")
        self._pending_calls = []

    def _resolve_pending_inherits(self) -> None:
        """Turn queued supertype references into class-to-class `inherits` edges.

        ARCHITECTURE.md has always listed `inherits` as a supported
        relationship, but no edge of that type was ever created. They are added
        here because the same supertype index that resolves an inherited call
        makes them free, and because they are the other genuinely cross-file
        structural relationship in these codebases.
        """
        for class_node_id, supertype_name in self._pending_inherits:
            candidates = self._class_nodes_by_name.get(supertype_name, [])
            if len(candidates) != 1:
                if candidates:
                    # Several classes share this simple name; picking one would
                    # be a guess.
                    reason = "inherits_ambiguous"
                elif supertype_name in self.call_resolver.files_by_class:
                    # The supertype IS declared in this corpus, but as an
                    # interface or enum - and only ClassDeclaration becomes a
                    # CPG node, so there is no node to point the edge at. This
                    # is the dominant case in Spring-style repos, where almost
                    # every supertype is an interface: `inherits` edges
                    # therefore under-report real structure until interfaces
                    # become first-class nodes. Counted separately so that gap
                    # is visible rather than looking like an external base class.
                    reason = "inherits_supertype_not_a_node"
                else:
                    reason = "inherits_outside_corpus"
                self.call_resolution_stats[reason] += 1
                continue
            target = candidates[0]
            if target == class_node_id:
                continue
            self.edges.append(CodeEdge(
                source_id=class_node_id,
                target_id=target,
                edge_type="inherits",
                metadata={"supertype": supertype_name},
            ))
        self._pending_inherits = []
    
    def _extract_source_text(self, content: str, node: Any) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Full declaration source for a class/method node, plus its extent.

        This is what Stage 3 should hand the generator: a single line (all
        `_extract_snippet` can give) is not a refactorable unit, and the
        transformer was trained on whole method bodies. Returns ("", None) when
        the extent cannot be determined, so callers fall back to code_snippet
        rather than acting on a guess.
        """
        try:
            extent = member_extent(content, node)
        except Exception as e:  # defensive: never let extent finding break CPG building
            logger.debug(f"Could not compute extent for {getattr(node, 'name', '?')}: {e}")
            return "", None
        if extent is None:
            return "", None
        return extent.text(content), extent.to_dict()

    def _extract_snippet(self, content: str, position: Tuple[int, int]) -> str:
        """Extract a ONE-LINE code snippet from file.

        Deliberately single-line: smell_detector's duplicate-code Jaccard
        (threshold 0.7) and boolean-operator counts (threshold 3) are calibrated
        on per-statement single lines. For the full declaration source use
        _extract_source_text / CodeNode.source_text instead.
        """
        if not position:
            return ""
        line = position[0]
        lines = content.split('\n')
        if line - 1 < len(lines):
            return lines[line - 1]
        return ""
    
    def get_graph(self) -> nx.MultiDiGraph:
        """Return the NetworkX graph"""
        return self.graph
    
    def get_node(self, node_id: str) -> Optional[CodeNode]:
        """Get a specific node"""
        return self.nodes.get(node_id)
    
    def get_dependencies(self, node_id: str, direction: str = "both") -> Set[str]:
        """
        Get dependent nodes
        direction: "in" (predecessors), "out" (successors), "both"
        """
        if direction == "in":
            return set(self.graph.predecessors(node_id))
        elif direction == "out":
            return set(self.graph.successors(node_id))
        else:
            predecessors = set(self.graph.predecessors(node_id))
            successors = set(self.graph.successors(node_id))
            return predecessors.union(successors)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize CPG to dictionary"""
        return {
            "nodes": {node_id: {
                "type": node.node_type,
                "name": node.name,
                "file": node.file_path,
                "line": node.line_number,
                "code_snippet": node.code_snippet,
                "source_text": node.source_text,
                "metadata": node.metadata
            } for node_id, node in self.nodes.items()},
            "edges": [
                {
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "type": edge.edge_type,
                    "metadata": edge.metadata
                } for edge in self.edges
            ]
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get CPG statistics.

        `graph_nodes` is reported alongside `total_nodes` as a consistency
        check: the two must be equal, and any gap means edges are referencing
        nodes that do not exist (which NetworkX would materialise silently).
        """
        edge_types = self._count_by_type(self.edges, "edge_type")
        call_edges = [e for e in self.edges if e.edge_type == "calls"]
        cross_file_calls = sum(1 for e in call_edges if e.metadata.get("cross_file"))
        attempted = len(call_edges) + len(self.unresolved_calls)
        return {
            "total_nodes": len(self.nodes),
            "graph_nodes": self.graph.number_of_nodes(),
            "total_edges": len(self.edges),
            "node_types": self._count_by_type(self.nodes.values(), "node_type"),
            "edge_types": edge_types,
            "data_flow_edges": edge_types.get("data_flow", 0),
            "control_flow_edges": edge_types.get("control_flow", 0),
            "call_graph_edges": edge_types.get("calls", 0),
            "inherits_edges": edge_types.get("inherits", 0),
            "cross_file_call_edges": cross_file_calls,
            "unresolved_calls": len(self.unresolved_calls),
            # Honest coverage: resolved / every invocation we saw. Library calls
            # and Lombok-generated members are legitimately unresolvable, so
            # this is NOT expected to approach 1.0 - see
            # src/cpg/call_resolver.py's OUT_OF_SCOPE_REASONS.
            "call_resolution_rate": (len(call_edges) / attempted) if attempted else 0.0,
            # Both rates together: the second excludes invocations no
            # source-only resolver could decide, so it reflects resolver
            # quality rather than corpus composition. Reported as a pair.
            "call_coverage": coverage(self.call_resolution_stats),
            "call_resolution_strategies": dict(self.call_resolution_stats),
            "total_files": len(self.file_map),
            "include_data_flow": self.include_data_flow,
            "include_control_flow": self.include_control_flow,
            "include_call_graph": self.include_call_graph
        }
    
    def get_visualization_data(self, max_nodes: int = 150) -> Dict[str, Any]:
        """
        Return a size-capped node/edge payload suitable for frontend graph rendering.
        Prioritizes structural nodes (class/method/field) over statement/variable detail
        so large codebases still produce a readable graph.
        """
        priority_order = {"class": 0, "method": 1, "field": 2, "function": 3, "variable": 4, "statement": 5}
        ordered_nodes = sorted(
            self.nodes.values(),
            key=lambda n: (priority_order.get(n.node_type, 99), n.file_path, n.line_number)
        )[:max_nodes]
        included_ids = {node.node_id for node in ordered_nodes}
        
        vis_nodes = [
            {
                "id": node.node_id,
                "label": node.name or node.node_type,
                "type": node.node_type,
                "file": node.file_path,
                "line": node.line_number
            }
            for node in ordered_nodes
        ]
        vis_edges = [
            {
                "source": edge.source_id,
                "target": edge.target_id,
                "type": edge.edge_type
            }
            for edge in self.edges
            if edge.source_id in included_ids and edge.target_id in included_ids
        ]
        
        return {
            "nodes": vis_nodes,
            "edges": vis_edges,
            "truncated": len(self.nodes) > max_nodes,
            "total_nodes": len(self.nodes),
            "shown_nodes": len(vis_nodes)
        }
    
    @staticmethod
    def _count_by_type(items, attr: str) -> Dict[str, int]:
        """Count items by attribute"""
        counts = defaultdict(int)
        for item in items:
            counts[getattr(item, attr)] += 1
        return dict(counts)
