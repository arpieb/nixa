defmodule NixaTest.Neighbors do
  use ExUnit.Case
  doctest Nixa

  test "KDTree" do
    num_samples = 101
    num_f = 3
    inputs = Nx.random_uniform({num_samples, num_f}) |> Nx.to_batched_list(1)
    tree = Nixa.Neighbors.KDTree.create(inputs)
    outputs = Nixa.Neighbors.KDTree.query(tree, inputs)
    num_matches = Nx.equal(Nx.concatenate(inputs), Nx.concatenate(outputs))
      |> Nx.sum()
      |> Nx.to_scalar()
    assert(num_matches == num_f * num_samples)
  end
end
