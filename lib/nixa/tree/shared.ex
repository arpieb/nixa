defmodule Nixa.Tree.Shared do
  @moduledoc """
  Functions common to tree construction
  """

  import Nixa.Stats
  import Nixa.Shared

  def get_split_vals(inputs, split_a) do
    inputs
      |> Enum.map(fn t -> t[[0..-1, split_a]] end)
      |> MapSet.new()
      |> MapSet.to_list()
  end

  def filter_inputs_targets(inputs, targets, split_a, split_val) do
    inputs
      |> Enum.zip(targets)
      |> Enum.filter(fn {i, _t} -> i[0][split_a] == split_val[0] end)
      |> Enum.unzip()
  end

  def calc_targets_entropy(targets) do
    targets
    |> Nx.concatenate()
    |> frequencies()
    |> prob_dist()
    |> entropy()
  end

  def calc_targets_gini_impurity(targets) do
    targets
    |> Nx.concatenate()
    |> frequencies()
    |> prob_dist()
    |> gini_impurity()
  end

  def calc_info_gain(inputs, targets, split_a, h) do
    get_split_vals(inputs, split_a)
    |> Enum.map(fn split_val -> calc_attr_entropy(inputs, targets, split_a, split_val) end)
    |> Enum.reduce(Nx.tensor(0.0), fn ha, acc -> Nx.add(ha, acc) end)
    |> Nx.negate()
    |> Nx.add(h)
  end

  def calc_gini_impurity(inputs, targets, split_a) do
    get_split_vals(inputs, split_a)
    |> Enum.map(fn split_val -> calc_attr_gini_impurity(inputs, targets, split_a, split_val) end)
    |> Enum.reduce(Nx.tensor(0.0), fn ga, acc -> Nx.add(ga, acc) end)
  end

  def calc_attr_entropy(inputs, targets, split_a, split_val) do
    {_v_inputs, v_targets} = filter_inputs_targets(inputs, targets, split_a, split_val)
    calc_targets_entropy(v_targets) |> Nx.multiply(Enum.count(v_targets)) |> Nx.divide(Enum.count(targets))
  end

  def calc_attr_gini_impurity(inputs, targets, split_a, split_val) do
    {_v_inputs, v_targets} = filter_inputs_targets(inputs, targets, split_a, split_val)
    calc_targets_gini_impurity(v_targets) |> Nx.multiply(Enum.count(v_targets)) |> Nx.divide(Enum.count(targets))
  end

  def calc_binning(inputs, binning_strategy) do
    t_inputs = Nx.concatenate(inputs)
    num_attrs = t_inputs |> Nx.shape() |> elem(1)
    binning_strategy = if is_tuple(binning_strategy),
      do: List.duplicate(binning_strategy, num_attrs),
      else: binning_strategy

    features = for idx <- 0..(num_attrs - 1), do: t_inputs[[0..-1, idx]]
    binning_borders = features
    |> Enum.zip(binning_strategy)
    |> Enum.map(fn {values, {binner, nbins}} -> apply(Nixa.Discretizers, binner, [values, nbins]) end)

    {binning_strategy, binning_borders}
  end

end
