defmodule Nixa.Tree.ID3Regressor do
  @moduledoc """
  Implementation of ID3 regressor decision tree algorithm
  """

  import Nixa.Tree.Shared
  import Nixa.Shared

  defmodule WrappedModel do
    defstruct [
      root: nil,
      binning_strategy: [],
      binning_borders: []
    ]
  end

  @doc """
  Train a model using the provided inputs and targets
  """
  def fit(inputs, targets, opts \\ []) when is_list(inputs) and is_list(targets) do
    binning_strategy = {:uniform, 10} #Keyword.get(opts, :binning_strategy, {:uniform, 10})
    {binning_strategy, binning_borders} = calc_binning(inputs, binning_strategy)
    xform_inputs = inputs
    |> Enum.map(fn inst -> Nixa.Discretizers.transform_instance(inst, binning_borders) end)

    num_attrs = inputs |> Enum.fetch!(0) |> Nx.size()
    root = build_tree({xform_inputs, targets}, MapSet.new(0..(num_attrs - 1)), opts)

    %WrappedModel{
      root: root,
      binning_strategy: binning_strategy,
      binning_borders: binning_borders
    }
  end

  @doc """
  Predict a value using a trained model
  """
  def predict(%WrappedModel{} = wrapped_model, inputs) when is_list(inputs) do
    model = wrapped_model.root
    binning_borders = wrapped_model.binning_borders

    inputs
    |> Enum.map(fn inst -> Nixa.Discretizers.transform_instance(inst, binning_borders) end)
    |> Enum.map(fn i -> traverse_tree(model, i) end)
  end

  ### Internal functions

  defp traverse_tree(node, input) do
    cond do
      node.children != nil and !Enum.empty?(node.children) ->
        cond do
          node.attr == nil ->
            nil
          true ->
            attr_val = input[0][node.attr] |> Nx.to_scalar()
            child = Map.get(node.children, attr_val)
            if child == nil, do: node.target, else: traverse_tree(child, input)
        end
      node.target != nil -> node.target
      true ->
        nil
    end
  end

  defp build_tree({inputs, targets}, attrs, opts) do
    h = calc_targets_entropy(targets)
    if Nx.to_scalar(h) == 0.0 do
      # Base case where there is only one target value
      t = Nx.concatenate(targets)
      %Nixa.Tree.Node{target: t[0]}
    else
      # Find split attribute
      split_arg = attrs
        |> Enum.map(fn a -> Task.async(fn -> calc_info_gain(inputs, targets, a, h) |> Nx.new_axis(0) end) end)
        |> Task.await_many(:infinity)
        |> Nx.concatenate()
        |> Nx.argmax()
        |> Nx.to_scalar()
      split_a = Enum.fetch!(attrs, split_arg)

      rem_attrs = MapSet.delete(attrs, split_a)

      split_vals = get_split_vals(inputs, split_a)
      children = split_vals
        |> Enum.map(fn val -> Task.async(fn -> {Nx.to_scalar(val[0]), create_child(inputs, targets, split_a, val, rem_attrs, opts)} end) end)
        |> Task.await_many(:infinity)
        |> Map.new()

      %Nixa.Tree.Node{
        attr: split_a,
        children: children,
        target: get_mean_target(targets)
      }
    end
  end

  defp create_child(inputs, targets, split_a, split_val, rem_attrs, opts) do
    {v_inputs, v_targets} = filter_inputs_targets(inputs, targets, split_a, split_val)
    cond do
      Enum.empty?(v_inputs) or Enum.empty?(v_targets) ->
        %Nixa.Tree.Node{
          target: get_mean_target(targets)
        }
      MapSet.size(rem_attrs) == 0 ->
        %Nixa.Tree.Node{
          target: get_mean_target(v_targets)
        }
      true -> build_tree({v_inputs, v_targets}, rem_attrs, opts)
    end
  end

end
