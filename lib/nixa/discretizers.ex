defmodule Nixa.Discretizers do
  @moduledoc """
  Provides feature discretizers
  """

  import Nixa.Shared

  @doc """
  Implements uniform binning approach
  """
  def uniform(%Nx.Tensor{} = values, nbins) do
    start = Nx.reduce_min(values)
    stop = Nx.reduce_max(values)
    Nixa.Shared.linspace(start, stop, nbins - 1)
    |> safe_to_tensor(type: Nx.type(values))
  end

  @doc """
  Transforms data instance based on provided binning borders
  """
  def transform_instance(values, borders) do
    batched_values = values
    |> Nx.squeeze()
    |> Nx.to_batched_list(1)

    borders
    |> Enum.zip(batched_values)
    |> Enum.map(fn {val_borders, value} -> transform_value(value, val_borders) end)
    |> Nx.tensor()
    |> Nx.new_axis(0)
  end

  @doc """
  Transforms single value based on provided binning borders
  """
  def transform_value(value, borders) do
    idx = Nx.less(value, borders)
    |> Nx.to_flat_list()
    |> Enum.find_index(fn x -> x == 1 end)

    if idx == nil, do: Nx.size(borders), else: idx
  end

end
