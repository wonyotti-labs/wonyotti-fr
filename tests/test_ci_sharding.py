import runpy

import pytest

runner = runpy.run_path('scripts/run_test_shard.py')
partition_nodes = runner['partition_nodes']
ShardSelection = runner['ShardSelection']


def test_all_cases_once_same_file_together_and_collection_order_independent():
    nodes = [(f'tests/test_{i}.py::test_value[{j}]', f'tests/test_{i}.py')
        for i in range(25) for j in range(i + 1)]
    groups = partition_nodes(nodes, 4)
    flattened = [node for group in groups for node in group]
    assert sorted(flattened) == sorted(node for node, _ in nodes)
    assert len(set(flattened)) == len(nodes)
    assert all(groups)
    owners = {node: index for index, group in enumerate(groups) for node in group}
    for filename in {filename for _, filename in nodes}:
        assert len({owners[node] for node, path in nodes if path == filename}) == 1
    reverse = partition_nodes(list(reversed(nodes)), 4)
    assert [set(group) for group in reverse] == [set(group) for group in groups]


@pytest.mark.parametrize('nodes,count', [([], 4), ([('n', 'tests/t.py')], 0),
    ([('n', 'tests/t.py'), ('n', 'tests/u.py')], 4), ([('n', '../t.py')], 4),
    ([('n', '/tmp/t.py')], 4)])
def test_invalid_collection_cannot_silently_omit_cases(nodes, count):
    with pytest.raises(ValueError):
        partition_nodes(nodes, count)


def test_invalid_shard_number_rejected_before_pytest(tmp_path):
    for index, count in [(-1, 4), (4, 4), (0, 0), (True, 4)]:
        with pytest.raises(ValueError):
            ShardSelection(index, count, tmp_path/'manifest.json')
