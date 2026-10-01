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
from collections import defaultdict
from loguru import logger

import javalang

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
    
    def _process_java_file(self, file_path: str, content: str) -> None:
        """Extract entities and relationships from Java file"""
        try:
            tree = javalang.parse.parse(content)
            
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
                # Build control-flow, data-flow and call edges for this method body
                self._process_method_body(file_path, content, method_node_id, node)
            
            # Extract field definitions
            for path, node in tree.filter(javalang.tree.FieldDeclaration):
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
        
        # Call graph: every invocation inside this method is attributed to it
        if self.include_call_graph:
            try:
                for _, invocation in method_node.filter(javalang.tree.MethodInvocation):
                    self.edges.append(CodeEdge(
                        source_id=method_node_id,
                        target_id=f"method_{file_path}_{invocation.member}",
                        edge_type="calls",
                        metadata={"qualifier": str(invocation.qualifier) if invocation.qualifier else ""}
                    ))
            except Exception as e:
                logger.debug(f"Error extracting method calls for {method_node_id}: {e}")
    
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
        """Build dependencies between files"""
        logger.info("Building cross-file dependencies...")
        
        for edge in self.edges:
            self.graph.add_edge(edge.source_id, edge.target_id, 
                              type=edge.edge_type, **edge.metadata)
    
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
        """Get CPG statistics"""
        edge_types = self._count_by_type(self.edges, "edge_type")
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": self._count_by_type(self.nodes.values(), "node_type"),
            "edge_types": edge_types,
            "data_flow_edges": edge_types.get("data_flow", 0),
            "control_flow_edges": edge_types.get("control_flow", 0),
            "call_graph_edges": edge_types.get("calls", 0),
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
