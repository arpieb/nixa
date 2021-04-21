defmodule Nixa.Shared do
  @moduledoc """
  Utility functions
  """

  import Nx.Defn

  @empty_mapset MapSet.new()

  defguard empty_mapset?(some_map_set) when some_map_set == @empty_mapset

  defn log2(x) do
    Nx.log(x) / Nx.log(2.0)
  end

  defn log10(x) do
    Nx.log(x) / Nx.log(10.0)
  end

 def get_categories(vals) do
  vals = if is_list(Enum.fetch!(vals, 0)), do: vals, else: Enum.map(vals, fn v -> [v] end)
  cats = vals
    |> Enum.fetch!(0)
    |> Enum.map(fn _ -> MapSet.new() end)
  vals
    |> Enum.reduce(cats, fn i, acc ->  update_categories_mapset(i, acc) end)
    |> Enum.map(fn ms -> MapSet.to_list(ms) end)
 end

 defp update_categories_mapset(i, acc) do
  acc
  |> Enum.zip(i)
  |> Enum.map(fn {s, val} -> MapSet.put(s, val) end)
 end

 def categorical_to_numeric([], _cats), do: []
 def categorical_to_numeric([input | rest], cats) do
   numval = cats
   |> Enum.zip(input)
   |> Enum.map(fn {cs, val} -> Enum.find_index(cs, fn x -> x == val end) end)
   [numval | categorical_to_numeric(rest, cats)]
 end

end
