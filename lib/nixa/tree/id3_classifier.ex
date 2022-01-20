defmodule Nixa.Tree.ID3Classifier do
  @moduledoc """
  Implementation of ID3 classifier decision tree algorithm
  """

  import Nixa.Tree.Shared
  import Nixa.Shared

  @doc """
  Train a model using the provided inputs and targets
  """
  def fit(inputs, targets, opts \\ []) when is_list(inputs) and is_list(targets) do
    num_attrs = inputs |> Enum.fetch!(0) |> Nx.size()
    build_tree({inputs, targets}, MapSet.new(0..(num_attrs - 1)), opts)
  end

  @doc """
  Predict classes using a trained model
  """
  def predict(%Nixa.Tree.Node{} = model, inputs) when is_list(inputs) do
    inputs
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
            attr_val = input[0][node.attr] |> Nx.to_number()
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
    if Nx.to_number(h) == 0.0 do
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
        |> Nx.to_number()
      split_a = Enum.fetch!(attrs, split_arg)

      rem_attrs = MapSet.delete(attrs, split_a)

      split_vals = get_split_vals(inputs, split_a)
      children = split_vals
        |> Enum.map(fn val -> {Nx.to_number(val[0]), create_child(inputs, targets, split_a, val, rem_attrs, opts)} end)
        |> Map.new()

      %Nixa.Tree.Node{
        attr: split_a,
        children: children,
        target: get_argmax_target(targets)
      }
    end
  end

  defp create_child(inputs, targets, split_a, split_val, rem_attrs, opts) do
    {v_inputs, v_targets} = filter_inputs_targets(inputs, targets, split_a, split_val)
    cond do
      Enum.empty?(v_inputs) or Enum.empty?(v_targets) ->
        %Nixa.Tree.Node{
          target: get_argmax_target(targets)
        }
      MapSet.size(rem_attrs) == 0 ->
        %Nixa.Tree.Node{
          target: get_argmax_target(v_targets)
        }
      true -> build_tree({v_inputs, v_targets}, rem_attrs, opts)
    end
  end

end
