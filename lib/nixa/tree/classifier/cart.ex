defmodule Nixa.Tree.Classifier.CART do
  @moduledoc """
  Implementation of ID3 classifier decision tree algorithm
  """

  import Nixa.Tree.Shared

  defmodule Node do
    import Kernel, except: [to_string: 1]

    defstruct [
      attr: nil,
      children: [],
      target: nil
    ]

    defimpl Inspect do
      def inspect(node, _opts) do
        Kernel.to_string(node)
      end
    end

    defimpl String.Chars do
      def to_string(node), do: node_to_string(node)

      def node_to_string(node, split_val \\ nil, level \\ 0) do
        prefix = List.duplicate(" ", level) |> Enum.join()
        out = prefix <> "node"

        out = out <> cond do
          split_val != nil -> ", split_val: #{split_val}"
          true -> ""
        end

        out = out <> cond do
          node.attr != nil -> ", split_attr: #{node.attr}"
          true -> ""
        end

        out = out <> cond do
          node.target != nil ->
            target = node.target[0] |> Nx.to_scalar()
            ", default_target: #{target}"
          true -> ""
        end

        out = out <> "\n"

        out <> cond do
          node.children != nil ->
            node.children
            |> Enum.reduce("", fn {attr_val, node}, acc -> acc <> node_to_string(node, attr_val, level + 2) end)
          true -> ""
        end
      end

    end

  end

  def fit(inputs, targets, opts \\ []) when is_list(inputs) and is_list(targets) do
    num_attrs = inputs |> Enum.fetch!(0) |> Nx.size()
    build_tree({inputs, targets}, MapSet.new(0..(num_attrs - 1)), opts)
  end

  def predict(%Node{} = model, inputs) when is_list(inputs) do
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
            attr_val = input[0][node.attr] |> Nx.to_scalar()
            child = Map.get(node.children, attr_val)
            cond do
              child == nil ->
                node.target
              true -> traverse_tree(child, input)
            end
        end
      node.target != nil -> node.target
      true ->
        nil
    end
  end

  defp build_tree({inputs, targets}, attrs, opts) do
    g = calc_targets_gini_impurity(targets)
    if Nx.to_scalar(g) == 0.0 do
      # Base case where there is only one target value
      t = Nx.concatenate(targets)
      %Node{target: t[0]}
    else
      # Find split attribute
      split_arg = attrs
        |> Enum.map(fn a -> Task.async(fn -> calc_gini_impurity(inputs, targets, a) |> Nx.new_axis(0) end) end)
        |> Task.await_many(:infinity)
        |> Nx.concatenate()
        |> Nx.argmin()
        |> Nx.to_scalar()
      split_a = Enum.fetch!(attrs, split_arg)

      rem_attrs = MapSet.delete(attrs, split_a)

      split_vals = get_split_vals(inputs, split_a)
      children = split_vals
        |> Enum.map(fn val -> Task.async(fn -> {Nx.to_scalar(val[0]), create_child(inputs, targets, split_a, val, rem_attrs, opts)} end) end)
        |> Task.await_many(:infinity)
        |> Map.new()

      %Node{
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
        %Node{
          target: get_argmax_target(targets)
        }
      MapSet.size(rem_attrs) == 0 ->
        %Node{
          target: get_argmax_target(v_targets)
        }
      true -> build_tree({v_inputs, v_targets}, rem_attrs, opts)
    end
  end

end
