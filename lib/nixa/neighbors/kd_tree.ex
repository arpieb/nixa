defmodule Nixa.Neighbors.KDTree do
  @moduledoc """
  Implements a BallTree algorithm
  """

  import Nixa.Shared

  defstruct [
    metric_fn: nil,
    metric_fn_args: nil,
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

      defp node_to_string(node, label, level \\ 0) do
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
  Create new KDTree from provided inputs using provided options
  """
  def create(inputs, opts \\ []) when is_list(inputs) do
    metric_fn = Keyword.get(opts, :metric, {:minkowski, [2]})
    metric_fn = if is_tuple(metric_fn), do: metric_fn, else: {metric_fn, []}
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
    |> Enum.map(fn input -> query_one(tree, input, opts) end)
  end

  ### Internal functions

  defp query_one(tree, input, opts) do
    traverse_tree(tree.root, tree.num_dims, tree.metric_fn, Nx.squeeze(input), 0, opts)
    |> Nx.new_axis(0)
  end

  defp traverse_tree(node, num_dims, metric_fn, input, depth, opts) do
    # TODO implement k-nearest logic
    # k = Keyword.get(opts, :k, 1)
    f = rem(depth, num_dims)
    {side, best} = cond do
      comp_tensors(input[f], node.location[f], &Nx.less/2) and node.left ->
        {:left, traverse_tree(node.left, num_dims, metric_fn, input, depth + 1, opts)}
      comp_tensors(input[f], node.location[f], &Nx.greater/2) and node.right ->
        {:right, traverse_tree(node.right, num_dims, metric_fn, input, depth + 1, opts)}
      true -> {:nil, node.location}
    end
    if !side do
      best
    else
      node_input_d = calc_metric(metric_fn, node.location, input)
      best_input_d = calc_metric(metric_fn, best, input)
      {best_dist, best} = if node_input_d < best_input_d, do: {node_input_d, node.location}, else: {best_input_d, best}
      if calc_metric(metric_fn, input[f], node.location[f]) < best_dist do
        poss_best = cond do
          side == :left and node.right -> traverse_tree(node.right, num_dims, metric_fn, input, depth + 1, opts)
          side == :right and node.left -> traverse_tree(node.left, num_dims, metric_fn, input, depth + 1, opts)
          true -> best
        end
        if calc_metric(metric_fn, poss_best, input) < best_dist, do: poss_best, else: best
      else
        best
      end
    end
  end

  defp comp_tensors(a, b, comp_fn) do
    (comp_fn.(a, b) |> Nx.to_scalar()) == 1
  end

  defp calc_metric({f, args}, x, y) do
    apply(Nixa.Distance, f, [x, y] ++ args)
  end

  defp kdtree([], _k, _depth), do: nil
  defp kdtree(inputs, num_dims, depth) do
    f = rem(depth, num_dims)
    num_inputs = Enum.count(inputs)
    inputs = Enum.sort(inputs, fn a, b -> Nx.to_scalar(a[0][f]) <= Nx.to_scalar(b[0][f]) end)
    median = div(num_inputs, 2)
    location = Enum.at(inputs, median) |> Nx.squeeze()
    {left, [_ | right]} = Enum.split(inputs, median)
    %Node{
      location: location,
      left: kdtree(left, num_dims, depth + 1),
      right: kdtree(right, num_dims, depth + 1),
    }
  end

end
