defmodule NixaTest.Discretizers do
  use ExUnit.Case
  doctest Nixa

  import Nixa.Tree.Shared

  test "uniform" do
    values = Nx.tensor([1, 3, 4, 4, 0, 2, 6, 6, 1, 0])
    assert Nixa.Discretizers.uniform(values, 5) == Nx.tensor([0.0, 2.0, 4.0, 6.0])
  end

  test "transform_value" do
    values = Nx.iota({10})
    borders = Nixa.Discretizers.uniform(values, 5)
    assert Nixa.Discretizers.transform_value(values[4], borders) == 2
    assert Nixa.Discretizers.transform_value(-1, borders) == 0
    assert Nixa.Discretizers.transform_value(100, borders) == 4
  end

  test "transform_instance" do
    inputs = Nx.iota({10, 2}) |> Nx.to_batched_list(1)
    binning_strategy = {:uniform, 10}
    {_binning_strategy, binning_borders} = calc_binning(inputs, binning_strategy)
    xform_inputs = inputs
      |> Enum.map(fn inst -> Nixa.Discretizers.transform_instance(inst, binning_borders) end)

    test_xform_inputs = xform_inputs
      |> Nx.concatenate()
      |> Nx.to_flat_list()
    assert test_xform_inputs == [1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]
  end

end
