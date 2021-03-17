defmodule Nixa.Kernels do
  @moduledoc false

  import Nx.Defn

  def distance(p1, p2), do: distance_euclidean(p1, p2)

  defn distance_euclidean(p1, p2) do
    Nx.subtract(p1, p2) |> Nx.power(2) |> Nx.sum() |> Nx.sqrt()
  end

  defn distance_manhattan(p1, p2) do
    Nx.subtract(p1, p2) |> Nx.abs() |> Nx.sum()
  end

end
