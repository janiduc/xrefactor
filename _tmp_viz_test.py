import tempfile, os, json
from src.cpg.cpg_builder import CodePropertyGraph

sample = '''
public class Foo {
    public int compute(int a, int b) {
        int sum = a + b;
        if (sum > 10) {
            sum = sum - 1;
            helper(sum);
        } else {
            sum = sum + 1;
        }
        for (int i = 0; i < sum; i++) {
            helper(i);
        }
        return sum;
    }

    public void helper(int x) {
        System.out.println(x);
    }
}
'''

d = tempfile.mkdtemp()
with open(os.path.join(d, 'Foo.java'), 'w') as f:
    f.write(sample)

cpg = CodePropertyGraph(language='java', include_data_flow=True, include_control_flow=True, include_call_graph=True)
cpg.build_from_directory(d)
viz = cpg.get_visualization_data(max_nodes=5)
print(json.dumps(viz, indent=2))
