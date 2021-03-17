defmodule Nixa.Optimize.SMO do
  @moduledoc ~S"""
  Implementation of the Sequential Minimal Optimization (SMO) algorithm as described in:
  Platt, J. (1998). Sequential minimal optimization: A fast algorithm for training support vector machines.
  """

  import Nx.Defn
  require Nixa.Shared

  @enforce_keys [:c, :tol, :kernel]
  defstruct [:c, :tol, :kernel, :kernel_opts, :alphas, :b, :sv_x, :sv_y]

  def create(kernel, kernel_opts \\ [], c \\ 1.0, tol \\ 0.001) do
    %__MODULE__{c: c, tol: tol, kernel: kernel, kernel_opts: kernel_opts, alphas: nil, b: 0.0, sv_x: nil, sv_y: nil}
  end

  def fit(model = %__MODULE__{}, inputs = %Nx.Tensor{}, targets = %Nx.Tensor{}, opts \\ []) do
    num_inputs = Nixa.Shared.axis_size(inputs, 0)
    max_iters = Keyword.get(opts, :max_iters) || 1
    model = %{model | alphas: Nx.broadcast(0.0, {num_inputs})}
    model = %{model | sv_x: inputs}
    model = %{model | sv_y: targets}
    train(model, inputs, targets, 0, true, max_iters)
  end

  def predict(%{kernel: kernel, alphas: alphas, b: b, sv_x: sv_x, sv_y: sv_y} = %__MODULE__{}, inputs = %Nx.Tensor{}) do
    inputs
    |> Nx.to_batched_list(1)
    |> Enum.map(&Nx.squeeze/1)
    |> Enum.map(fn x -> compute_k_vals(kernel, sv_x, x) |> predict_one(alphas, b, sv_y) |> Nx.new_axis(0) end)
    |> Nx.concatenate()
  end

  ##
  ##

  defnp predict_one(k_vals, alpha, b, sv_y) do
    sv_y |> Nx.multiply(alpha) |> Nx.multiply(k_vals) |> Nx.subtract(b) |> Nx.sum()
  end

  defp compute_k_vals(kernel, sv_x, x) do
    Nx.to_batched_list(sv_x, 1)
    |> Enum.map(&Nx.squeeze/1)
    |> Enum.map(fn x_j -> kernel.(x_j, x) |> Nx.new_axis(0) end)
    |> Nx.concatenate()
  end

  defp train(model, _inputs, _targets, _num_changed, _examine_all, 0), do: model
  defp train(model, _inputs, _targets, 0, false, _iters), do: model
  defp train(model, inputs, targets, num_changed, examine_all, iters) do
    num_inputs = Nixa.Shared.axis_size(inputs, 0)
    num_changed = 0
    if examine_all do
      {model, _, _, num_changed} = Enum.reduce(0..(num_inputs - 1), {model, inputs, targets, 0}, &reduce_examine_example/2)
    end

    train(model, inputs, targets, num_changed, examine_all, iters - 1)
  end

  def reduce_examine_example(i, {model, inputs, targets, count}) do
    {step_taken, model} = examine_example(model, i, inputs, targets) #|> IO.inspect(label: "train: examine_example: ")
    if step_taken, do: {model, inputs, targets, count + 1}, else: {model, inputs, targets, count}
  end

  defp examine_example(%{kernel: kernel, c: c, tol: tol, alphas: alphas, b: b, sv_x: sv_x, sv_y: sv_y} = model, i2, inputs, targets) do
    y2 = targets[i2] #|> IO.inspect(label: "examine_example: y2")
    y2_pred = compute_k_vals(kernel, sv_x, inputs[i2]) |> predict_one(alphas, b, sv_y)
    alph2 = alphas[i2]

    e2 = Nx.subtract(y2_pred, y2)
    r2 = Nx.multiply(e2, y2)
    if (r2 < tol && alph2 < c) || (r2 > tol && alph2 > 0.0) do
      num_good_alphas = Nx.reduce(alphas, 0, [type: {:s, 64}], fn a, acc -> if Nx.not_equal(a, 0.0) && Nx.not_equal(a, c), do: Nx.add(acc, 1), else: acc end)
      if Nx.greater(num_good_alphas, 1) do
        # Use heuristics
        i1 = 0 # todo i1 = result of second choice heuristic (section 2.2)
        {step_taken, model} = take_step(model, i1, i2, inputs, targets)
        if step_taken do
          {true, model}
        else
          # Brute force through all non-zero, non-C alphas
          num_alphas = Nixa.Shared.axis_size(alphas, 0)
          start = :rand.uniform(num_alphas) - 1
          {step_taken, model} = Enum.reduce_while(start..(start + num_alphas - 1), {false, model}, fn i, acc ->
            i1 = div(i, num_alphas)
            {step_taken, model} = take_step(model, i1, i2, inputs, targets)
            if alphas[i1] != 0.0 && alphas[i1] != c &&  step_taken do
              {:halt, {true, model}}
            else
              {:cont, acc}
            end
          end)
          if step_taken do
            {true, model}
          else
            # Brute force through all alphas
            start = :rand.uniform(num_alphas) - 1
            {step_taken, model} = Enum.reduce_while(start..(start + num_alphas - 1), {false, model}, fn i, {step_taken, model} ->
              i1 = div(i, num_alphas)
              {step_taken, model} = take_step(model, i1, i2, inputs, targets)
              if step_taken do
                {:halt, {true, model}}
              end
              {:cont, {false, model}}
            end)
            if step_taken, do: {true, model}, else: {false, model}
          end
        end
      end
    end

    # Fail spectacularly
    {false, model}
  end

  defp take_step(model, i1, i2, _inputs, _targets) when i1 == i2, do: {false, model}
  defp take_step(%{kernel: kernel, c: c, tol: _tol, alphas: alphas, b: b, sv_x: sv_x, sv_y: sv_y} = model, i1, i2, inputs, targets) do
    l = compute_l(c, alphas, i1, i2, targets)
    h = compute_h(c, alphas, i1, i2, targets)
    compute_updates(model, i1, i2, inputs, targets, l, h)
  end

  defp compute_updates(model, _i1, _i2, _inputs, _targets, l, h) when l == h, do: {false, model}
  defp compute_updates(%{kernel: kernel, c: _c, tol: _tol, alphas: alphas, b: b, sv_x: sv_x, sv_y: sv_y} = model, i1, i2, inputs, targets, l, h) do
    alph1 = alphas[i1]
    alph2 = alphas[i2]

    x1 = inputs[i1]
    x2 = inputs[i2]
    y1 = targets[i1]
    y2 = targets[i2]

    y1_pred = compute_k_vals(kernel, sv_x, x1) |> predict_one(alphas, b, sv_y)
    e1 = Nx.subtract(y1_pred, y1) # todo check in error cache
    y2_pred = compute_k_vals(kernel, sv_x, x2) |> predict_one(alphas, b, sv_y)
    e2 = Nx.subtract(y2_pred, y2) # todo check in error cache

    s = Nx.multiply(y1, y2)

    k11 = kernel.(x1, x1)
    k12 = kernel.(x1, x2)
    k22 = kernel.(x2, x2)
    eta = Nx.subtract(Nx.add(k11, k22), Nx.multiply(2.0, k12)) #k11 + k22 - 2.0 * k12

    a2 = 0.0
    eps = 0.0 # todo find def in paper

    if Nx.greater(eta, 0.0) do
      a2 = Nx.subtract(e1, e2) |> Nx.divide(eta) |> Nx.multiply(y2) |> Nx.add(alph2)
      a2 = cond do
        Nx.less(a2, l) -> l
        Nx.greater(a2, h) -> h
        true -> a2
      end
    else
      lobj = 0.0 # todo objective function at a2=L
      hobj = 0.0 # todo objective function at a2=H
      a2 = cond do
        Nx.less(lobj, Nx.subtract(hobj, eps)) -> l
        Nx.greater(lobj, Nx.add(hobj, eps)) -> h
        true -> alph2
      end
    end

    if Nx.less(Nx.abs(Nx.subtract(a2, alph2)), Nx.add(a2, alph2) |> Nx.add(eps) |> Nx.multiply(eps)) do
      {false, model}
    else
      a1 = Nx.add(alph1, Nx.multiply(s, Nx.subtract(alph2, a2))) #alph1 + s * (alph2 - a2)
      # todo Update threshold to reflect change in Lagrange multipliers
      # todo Update error cache using new Lagrange multipliers
      # Store a1, a2 in the alpha array
      num_alphas = Nixa.Shared.axis_size(alphas, 0)
      alphas = Nx.map(0..(num_alphas - 1), fn i ->
        cond do
          Nx.equal(i, i1) -> a1
          Nx.equal(i, i2) -> a2
          true -> alphas[i]
        end
      end)
      {true, %{model | alphas: alphas}}
    end
  end

  defnp compute_l(c, alphas, i1 \\ 0, i2 \\ 0, targets) do
    a1 = alphas[i1]
    a2 = alphas[i2]
    y1 = targets[i1]
    y2 = targets[i2]
    if Nx.not_equal(y1, y2) do
      Nx.max(0.0, Nx.subtract(a2, a1))
    else
      Nx.max(0.0, Nx.add(a2, a1) |> Nx.subtract(c))
    end
  end

  defnp compute_h(c, alphas, i1 \\ 0, i2 \\ 0, targets) do
    a1 = alphas[i1]
    a2 = alphas[i2]
    y1 = targets[i1]
    y2 = targets[i2]
    if Nx.not_equal(y1, y2) do
      Nx.min(c, Nx.add(c, a2) |> Nx.subtract(a1))
    else
      Nx.min(c, Nx.add(a2, a1))
    end
  end

end
