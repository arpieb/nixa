defmodule NixaTest.Shared do
  use ExUnit.Case
  doctest Nixa

  test "linspace" do
    endp_t = Nx.tensor([2.0, 2.25, 2.5, 2.75, 3.0])
    endp_f = Nx.tensor([2.0, 2.2, 2.4, 2.6, 2.8])
    assert Nixa.Shared.linspace(2, 3, 5) == endp_t
    assert Nixa.Shared.linspace(2, 3, 5, endpoint: false) == endp_f
  end

  test "get_categories" do
    {inputs, targets} = get_play_tennis_data()
    assert Nixa.Shared.get_categories(inputs) == [
      [:overcast, :rain, :sunny],
      [:cool, :hot, :mild],
      [:high, :normal],
      [:strong, :weak]
    ]
    assert Nixa.Shared.get_categories(targets) == [[:no, :yes]]
  end

  test "categorical_to_numeric" do
    {inputs, targets} = get_play_tennis_data()
    input_c = Nixa.Shared.get_categories(inputs)
    assert Nixa.Shared.categorical_to_numeric(inputs, input_c) == [
      [2, 1, 0, 1],
      [2, 1, 0, 0],
      [0, 1, 0, 1],
      [1, 2, 0, 1],
      [1, 0, 1, 1],
      [1, 0, 1, 0],
      [0, 0, 1, 0],
      [2, 2, 0, 1],
      [2, 0, 1, 1],
      [1, 2, 1, 1],
      [2, 2, 1, 0],
      [0, 2, 0, 0],
      [0, 1, 1, 1],
      [1, 2, 0, 0]
    ]
    target_c = Nixa.Shared.get_categories(targets)
    assert Nixa.Shared.categorical_to_numeric(targets, target_c) == [[0], [0], [1], [1], [1], [0], [1], [0], [1], [1], [1], [1], [1], [0]]
  end

  test "safe_to_tensor" do
    vals = [1, 2, 3]
    t = Nx.tensor(vals)
    assert Nixa.Shared.safe_to_tensor(t) == t
    assert Nixa.Shared.safe_to_tensor(vals) == t
  end

  test "safe_to_scalar" do
    val = 1
    t = Nx.tensor(val)
    assert Nixa.Shared.safe_to_scalar(t) == val
    assert Nixa.Shared.safe_to_scalar(val) == val
  end

  defp get_play_tennis_data() do
    [
      { [:sunny, :hot, :high, :weak], [:no] },
      { [:sunny, :hot, :high, :strong], [:no] },
      { [:overcast, :hot, :high, :weak], [:yes] },
      { [:rain, :mild, :high, :weak], [:yes] },
      { [:rain, :cool, :normal, :weak], [:yes] },
      { [:rain, :cool, :normal, :strong], [:no] },
      { [:overcast, :cool, :normal, :strong], [:yes] },
      { [:sunny, :mild, :high, :weak], [:no] },
      { [:sunny, :cool, :normal, :weak], [:yes] },
      { [:rain, :mild, :normal, :weak], [:yes] },
      { [:sunny, :mild, :normal, :strong], [:yes] },
      { [:overcast, :mild, :high, :strong], [:yes] },
      { [:overcast, :hot, :normal, :weak], [:yes] },
      { [:rain, :mild, :high, :strong], [:no] },
      ] |> Enum.unzip()

    # input_c = Nixa.Shared.get_categories(inputs)
    # x = Nixa.Shared.categorical_to_numeric(inputs, input_c) |> Nx.tensor() |> Nx.to_batched_list(1)

    # target_c = Nixa.Shared.get_categories(targets)
    # y = Nixa.Shared.categorical_to_numeric(targets, target_c) |> Nx.tensor() |> Nx.to_batched_list(1)
    # {x, y}
  end

end
