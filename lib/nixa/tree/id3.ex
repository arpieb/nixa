defmodule Nixa.Tree.ID3 do
  @moduledoc """
  Implementation of ID3 decision tree algorithm
  """

  import Nixa.Stats

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

  def fit(inputs, targets, opts \\ []) do
    num_attrs = inputs |> Enum.fetch!(0) |> Nx.size()
    build_tree({inputs, targets}, MapSet.new(0..(num_attrs - 1)), opts)
  end

  def predict(model, inputs) do
    inputs
    |> Enum.map(fn i -> traverse_tree(model, i) end)
  end

  ### Internal functions

  defp traverse_tree(nil, _input) do
    IO.puts("traverse_tree : node is nil")
    nil
  end

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
    h = calc_targets_entropy(targets)
    if Nx.to_scalar(h) == 0.0 do
      # Base case where there is only one target value
      t = Nx.concatenate(targets)
      %Node{target: t[0]}
    else
      expand_tree(inputs, targets, attrs, h, opts)
    end
  end

  defp expand_tree(inputs, targets, attrs, h, opts) do
    # Find split attribute
    split_arg = attrs
      |> Enum.map(fn a -> calc_info_gain(inputs, targets, a, h) |> Nx.to_scalar() end)
      |> Nx.tensor()
      |> Nx.argmax()
      |> Nx.to_scalar()
    split_a = Enum.fetch!(attrs, split_arg)

    rem_attrs = MapSet.delete(attrs, split_a)

    split_vals = get_split_vals(inputs, split_a)
    children = split_vals
      |> Enum.map(fn val -> {Nx.to_scalar(val[0]), create_child(inputs, targets, split_a, val, rem_attrs, opts)} end)#{Nx.to_scalar(val[0]), build_tree(filter_inputs_targets(inputs, targets, split_a, val), rem_attrs, opts)} end)
      |> Map.new()

    %Node{
      attr: split_a,
      children: children,
      target: get_max_target(targets)
    }
  end

  defp create_child(inputs, targets, split_a, split_val, rem_attrs, opts) do
    {v_inputs, v_targets} = filter_inputs_targets(inputs, targets, split_a, split_val)
    cond do
      Enum.empty?(v_inputs) or Enum.empty?(v_targets) ->
        %Node{
          target: get_max_target(targets)
        }
      MapSet.size(rem_attrs) == 0 ->
        %Node{
          target: get_max_target(v_targets)
        }
      true -> build_tree({v_inputs, v_targets}, rem_attrs, opts)
    end
  end

  defp get_split_vals(inputs, split_a) do
    inputs
      |> Enum.map(fn t -> t[[0..-1, split_a]] end)
      |> MapSet.new()
      |> MapSet.to_list()
  end

  defp get_max_target(targets) do
    target_idx = targets |> Nx.concatenate() |> frequencies() |> Nx.argmax() |> Nx.to_scalar()
    Enum.fetch!(targets, target_idx)[0]
  end

  defp filter_inputs_targets(inputs, targets, split_a, split_val) do
    inputs
      |> Enum.zip(targets)
      |> Enum.filter(fn {i, _t} -> i[0][split_a] == split_val[0] end)
      |> Enum.unzip()
  end

  defp calc_info_gain(inputs, targets, split_a, h) do
    # TODO

    get_split_vals(inputs, split_a)
    |> Enum.map(fn split_val -> calc_attr_entropy(inputs, targets, split_a, split_val) end)
    |> Enum.reduce(Nx.tensor(0.0), fn ha, acc -> Nx.add(ha, acc) end)
    |> Nx.negate()
    |> Nx.add(h)

    # :random.uniform()
  end

  defp calc_attr_entropy(inputs, targets, split_a, split_val) do
    {_v_inputs, v_targets} = filter_inputs_targets(inputs, targets, split_a, split_val)
    calc_targets_entropy(v_targets) |> Nx.multiply(Enum.count(v_targets)) |> Nx.divide(Enum.count(targets))
  end

  defp calc_targets_entropy(targets) do
    targets
    |> Nx.concatenate()
    |> frequencies()
    |> prob_dist()
    |> entropy()
  end

  defp frequencies(%Nx.Tensor{} = t) do
    classes = t |> Nx.to_flat_list() |> MapSet.new()
    classes
    |> MapSet.to_list()
    |> Nx.tensor()
    |> Nx.map(fn c -> Nx.equal(t, c) |> Nx.sum() end)
  end

end
