defmodule Nixa.Tree.Node do
  @moduledoc """
  Provides a common tree node for use in standard decision tree models
  """

  import Kernel, except: [to_string: 1]

  defstruct [
    attr: nil,
    children: [],
    target: nil
  ]

  defimpl Inspect do
    @moduledoc """
    Implements the Inspect protocol for Nixa.Tree.Node
    """
    def inspect(node, _opts) do
      Kernel.to_string(node)
    end
  end

  defimpl String.Chars do
    @moduledoc """
    Implements the String.Chars protocol for Nixa.Tree.Node
    """
    def to_string(node), do: node_to_string(node)

    defp node_to_string(node, split_val \\ nil, level \\ 0) do
      prefix = List.duplicate(" ", level) |> Enum.join()
      out = prefix <> "node"

      out = out <> if split_val != nil, do: ", split_val: #{split_val}", else: ""
      out = out <> if node.attr != nil, do: ", split_attr: #{node.attr}", else: ""
      out = out <> if node.target != nil, do: ", default_target: #{node.target[0] |> Nx.to_scalar()}", else: ""

      out = out <> "\n"

      out <> if node.children != nil,
        do: node.children |> Enum.reduce("", fn {attr_val, node}, acc -> acc <> node_to_string(node, attr_val, level + 2) end),
        else: ""
    end

  end

end
