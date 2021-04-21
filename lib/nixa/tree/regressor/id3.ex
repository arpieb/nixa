defmodule Nixa.Tree.Regressor.ID3 do
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
    # num_attrs = inputs |> Enum.fetch!(0) |> Nx.size()
    # build_tree({inputs, targets}, MapSet.new(0..(num_attrs - 1)), opts)
  end

  def predict(%Node{} = model, inputs) when is_list(inputs) do
    # inputs
    # |> Enum.map(fn i -> traverse_tree(model, i) end)
  end

end
