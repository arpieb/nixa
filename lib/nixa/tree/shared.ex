defmodule Nixa.Tree.Shared do
  @moduledoc """
  Functions common to tree construction
  """

  import Nixa.Stats

  def get_split_vals(inputs, split_a) do
    inputs
      |> Enum.map(fn t -> t[[0..-1, split_a]] end)
      |> MapSet.new()
      |> MapSet.to_list()
  end

  def get_argmax_target(targets) do
    target_idx = targets |> Nx.concatenate() |> frequencies() |> Nx.argmax() |> Nx.to_scalar()
    Enum.fetch!(targets, target_idx)[0]
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

  def frequencies(%Nx.Tensor{} = t) do
    t
    |> Nx.to_flat_list()
    |> MapSet.new()
    |> MapSet.to_list()
    |> Nx.tensor()
    |> Nx.map(fn c -> Nx.equal(t, c) |> Nx.sum() end)
  end

  def calc_info_gain(inputs, targets, split_a, h) do
    get_split_vals(inputs, split_a)
    |> Enum.map(fn split_val -> calc_attr_entropy(inputs, targets, split_a, split_val) end)
    |> Enum.reduce(Nx.tensor(0.0), fn ha, acc -> Nx.add(ha, acc) end)
    |> Nx.negate()
    |> Nx.add(h)
  end

  def calc_attr_entropy(inputs, targets, split_a, split_val) do
    {_v_inputs, v_targets} = filter_inputs_targets(inputs, targets, split_a, split_val)
    calc_targets_entropy(v_targets) |> Nx.multiply(Enum.count(v_targets)) |> Nx.divide(Enum.count(targets))
  end

end
