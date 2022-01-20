defmodule NixaTest.Distance do
  use ExUnit.Case
  doctest Nixa

  @delta 1.0e-6

  import Nixa.Distance

  test "euclidean" do
    {x, y} = get_xy()
    d = euclidean(x, y) |> Nx.to_number()
    assert_in_delta(d, 5.196152210235596, @delta)
  end

  test "manhattan" do
    {x, y} = get_xy()
    d = manhattan(x, y) |> Nx.to_number()
    assert d == 9
  end

  test "chebyshev" do
    {x, y} = get_xy()
    d = chebyshev(x, y) |> Nx.to_number()
    assert d == 3
  end

  test "minkowski" do
    {x, y} = get_xy()
    d = minkowski(x, y, 1.5) |> Nx.to_number()
    assert_in_delta(d, 6.24025147, @delta)
  end

  test "wminkowski" do
    {x, y} = get_xy()
    d = wminkowski(x, y, 1.5, 0.5) |> Nx.to_number()
    assert_in_delta(d, 3.12012573, @delta)
  end

  test "seuclidean" do
    {x, y} = get_xy()
    d = seuclidean(x, y, 0.5) |> Nx.to_number()
    assert_in_delta(d, 7.34846923, @delta)
  end

  test "hamming" do
    {x, y} = get_xy()
    d = hamming(x, y) |> Nx.to_number()
    assert d == 1
  end

  test "canberra" do
    {x, y} = get_xy()
    d = canberra(x, y) |> Nx.to_number()
    assert_in_delta(d, 2.02857143, @delta)
  end

  test "braycurtis" do
    {x, y} = get_xy()
    d = braycurtis(x, y) |> Nx.to_number()
    assert_in_delta(d, 0.6, @delta)
  end

  defp get_xy() do
    x = Nx.tensor([[0, 1, 2]])
    y = Nx.tensor([[3, 4, 5]])
    {x, y}
  end

end
