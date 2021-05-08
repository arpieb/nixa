defmodule Nixa.Distance do
  @moduledoc """
  Implements various distance metrics
  """

  import Nx.Defn

  defn euclidean(x, y) do
    Nx.sqrt(Nx.sum(Nx.power(x - y, 2)))
  end

  defn sqeuclidean(x, y) do
    Nx.sum(Nx.power(x - y, 2))
  end

  defn manhattan(x, y) do
    Nx.sum(Nx.abs(x - y))
  end

  defn chebyshev(x, y) do
    Nx.reduce_max(Nx.abs(x - y))
  end

  defn minkowski(x, y, p) do
    Nx.power(Nx.sum(Nx.power(Nx.abs(x - y), p)), 1 / p)
  end

  defn wminkowski(x, y, p, w) do
    Nx.power(Nx.sum(Nx.power(Nx.abs(w * (x - y)), p)), 1 / p)
  end

  defn seuclidean(x, y, v) do
    Nx.sqrt(Nx.sum(Nx.power(x - y, 2)) / v)
  end

  defn hamming(x, y) do
    Nx.select(Nx.not_equal(x, y), 1, 0) |> Nx.sum() |> Nx.divide(max(Nx.size(x), Nx.size(y)))
  end

  defn canberra(x, y) do
    Nx.sum(Nx.abs(x - y) / (Nx.abs(x) + Nx.abs(y)))
  end

  defn braycurtis(x, y) do
    Nx.sum(Nx.abs(x - y) / (Nx.sum(Nx.abs(x)) + Nx.sum(Nx.abs(y))))
  end

end
