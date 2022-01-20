defmodule Nixa.Shared do
  @moduledoc """
  Utility functions
  """

  import Nx.Defn

  @empty_mapset MapSet.new()

  @doc """
  Provide function guard to check for empty MapSet
  """
  defguard empty_mapset?(some_map_set) when some_map_set == @empty_mapset

  @doc """
  Provide simple log2 function
  """
  defn log2(x) do
    Nx.log(x) / Nx.log(2.0)
  end

  @doc """
  Provide simple log10 function
  """
  defn log10(x) do
    Nx.log(x) / Nx.log(10.0)
  end

  @doc """
  Given a set of values, identify all unique categories and return them
  """
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

  @doc """
  Given a set of values and the category mappings identified by get_categories/1, transform to numerical IDs
  """
  def categorical_to_numeric([], _cats), do: []
  def categorical_to_numeric([input | rest], cats) do
    numval = cats
      |> Enum.zip(input)
      |> Enum.map(fn {cs, val} -> Enum.find_index(cs, fn x -> x == val end) end)
      [numval | categorical_to_numeric(rest, cats)]
  end

  @doc """
  Provide basic linspace functionality, similar to numpy.linspace
  """
  def linspace(start, stop, num, opts \\ []) do
    start = safe_to_tensor(start)
    stop = safe_to_tensor(stop)
    num = safe_to_number(num)
    endpoint = Keyword.get(opts, :endpoint, true)
    step = if endpoint,
      do: calc_linspace_step_size(start, stop, Nx.subtract(num, 1)),
      else: calc_linspace_step_size(start, stop, num)
    values = for x <- 0..num-1, do: calc_linspace_step(start, step, x)
    Nx.stack(values)
  end

  defnp calc_linspace_step_size(start, stop, num) do
    Nx.abs(start - stop) / num
  end

  defnp calc_linspace_step(start, step, x) do
    start + step * x
  end

  @doc """
  Get the most common class from targets
  """
  def get_argmax_target(targets) do
    t_targets = targets |> Nx.concatenate()
    argmax = t_targets |> frequencies() |> Nx.argmax() |> Nx.to_number()
    t_targets
    |> Nx.to_flat_list()
    |> MapSet.new()
    |> MapSet.to_list()
    |> Enum.fetch!(argmax)
  end

  @doc """
  Get the mean value of all targets
  """
  def get_mean_target(targets) do
    targets |> Nx.concatenate() |> Nx.mean() |> Nx.new_axis(0)
  end

  @doc """
  Calculate the frequencies of values in a tensor
  """
  def frequencies(%Nx.Tensor{} = t) do
    t
    |> Nx.to_flat_list()
    |> MapSet.new()
    |> MapSet.to_list()
    |> Enum.map(fn c -> Nx.equal(t, c) |> Nx.sum() end)
    |> Nx.stack()
  end

  @doc """
  Provide Nx.tensor/2 functionality that is type-aware and can be blindly used on Nx.Tensor and scalars
  """
  def safe_to_tensor(%Nx.Tensor{} = t, _opts), do: t
  def safe_to_tensor(t, opts), do: Nx.tensor(t, opts)
  def safe_to_tensor(%Nx.Tensor{} = t), do: t
  def safe_to_tensor(t), do: Nx.tensor(t)

  @doc """
  Provide Nx.to_number/1 functionality that is type-aware and can be blindly used on Nx.Tensor and scalars
  """
  def safe_to_number(%Nx.Tensor{} = t), do: Nx.to_number(t)
  def safe_to_number(t), do: t

  def axis_size(%Nx.Tensor{} = t, axis), do: Nx.shape(t) |> elem(axis)

end
