defmodule Nixa.Neighbors.KDTree do
  @moduledoc """
  Implements a BallTree algorithm
  """

  import Nixa.Shared

  defstruct [
    metric_fn: nil,
    num_dims: nil,
    root: nil,
  ]

  defmodule Node do
    @moduledoc """
    Provides a binary tree node
    """

    defstruct [
      location: nil,
      left: nil,
      right: nil
    ]

    defimpl Inspect do
      @moduledoc """
      Implements the Inspect protocol for Nixa.Neighbors.KDTree.Node
      """
      def inspect(node, _opts) do
        Kernel.to_string(node)
      end
    end

    defimpl String.Chars do
      @moduledoc """
      Implements the String.Chars protocol for Nixa.Neighbors.KDTree.Node
      """
      def to_string(node), do: node_to_string(node, "root")

      defp node_to_string(node, label \\ "node", level \\ 0) do
        prefix = List.duplicate(" ", level) |> Enum.join()
        out = prefix <> label

        loc = node.location |> Nx.to_flat_list() |> Enum.map(&Kernel.to_string/1) |> Enum.join(", ")
        out = out <> ", location: " <> loc
        out = out <> "\n"

        out = out <> if node.left,
          do: node_to_string(node.left, "L", level + 2),
          else: ""
        out = out <> if node.right,
          do: node_to_string(node.right, "R", level + 2),
          else: ""

        out
      end

    end

  end

  @doc """
  Create new BallTree from provided inputs using provided options
  """
  def create(inputs, opts \\ []) when is_list(inputs) do
    metric_fn = Keyword.get(opts, :metric, :sqeuclidean)
    num_dims = inputs |> hd() |> axis_size(1)
    root = kdtree(inputs, num_dims, 0)

    %__MODULE__{
      metric_fn: metric_fn,
      root: root,
      num_dims: num_dims
    }
  end

  @doc """
  Query a tree for nearest neighbors to provided inputs
  """
  def query(tree, inputs, opts \\ []) do
    inputs
    |> Enum.map(fn input -> Task.async(fn -> query_one(tree, input, opts) end) end)
    |> Task.await_many()
  end

  ### Internal functions

  defp query_one(tree, input, opts) do
    # TODO implement k-nearest logic
    k = Keyword.get(opts, :k, 1)
    traverse_tree(tree.root, tree.num_dims, input, 0)
  end

  defp traverse_tree(node, num_dims, input, depth) do
    f = rem(depth, num_dims)
    cond do
      comp_tensors(input[0][f], node.location[f], &Nx.less/2) and node.left -> traverse_tree(node.left, num_dims, input, depth + 1)
      comp_tensors(input[0][f], node.location[f], &Nx.greater/2) and node.right -> traverse_tree(node.right, num_dims, input, depth + 1)
      true -> node.location
    end
  end

  defp comp_tensors(a, b, comp_fn) do
    (comp_fn.(a, b) |> Nx.to_scalar()) == 1
  end

  defp kdtree([], _k, _depth), do: nil
  defp kdtree(inputs, num_dims, depth) do
    f = rem(depth, num_dims)
    num_inputs = Enum.count(inputs)
    inputs = Enum.sort(inputs, fn a, b -> Nx.to_scalar(a[0][f]) <= Nx.to_scalar(b[0][f]) end)
    median = div(num_inputs, 2)
    location = Enum.at(inputs, median) |> Nx.squeeze()
    {left, [_ | right]} = Enum.split(inputs, median)
    left_task = Task.async(fn -> kdtree(left, num_dims, depth + 1) end)
    right_task = Task.async(fn -> kdtree(right, num_dims, depth + 1) end)
    [left_child, right_child] = Task.await_many([left_task, right_task])
    %Node{
      location: location,
      left: left_child, #kdtree(left, k, depth + 1),
      right: right_child #kdtree(right, k, depth + 1),
    }
  end

end
