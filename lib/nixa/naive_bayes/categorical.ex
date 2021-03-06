defmodule Nixa.NaiveBayes.Categorical do
  @moduledoc """
  Implements a categorical Naive Bayes classifier
  """

  import Nixa.Shared
  import Nixa.Stats
  import Nixa.NaiveBayes.Shared

  defstruct [
    class_probs: nil,
    feature_probs: nil,
    alpha: nil
  ]

  @doc """
  Train a model using the provided inputs and targets
  """
  def fit(inputs, targets, opts \\ []) do
    class_probability = Keyword.get(opts, :class_probability, :weighted)
    alpha = Keyword.get(opts, :alpha, 1.0)
    class_probs = if is_list(class_probability),
      do: class_probability,
      else: calc_class_prob(targets, class_probability, alpha)
    num_classes = class_probs |> Nx.size() |> Nx.to_scalar()
    feature_probs = 0..(num_classes - 1)
      |> Enum.map(fn c -> Task.async(fn -> calc_feature_probs(c, inputs, targets) end) end)
      |> Task.await_many(:infinity)

    %__MODULE__{
      class_probs: class_probs,
      feature_probs: feature_probs,
      alpha: alpha
    }
  end

  @doc """
  Predict classes using a trained model
  """
  def predict(%__MODULE__{} = model, inputs) do
    inputs
    |> Enum.map(fn input -> predict_one(model, input) end)
  end

  ### Internal functions

  defp predict_one(model, input) do
    model.class_probs
    |> Nx.to_flat_list()
    |> Enum.zip(model.feature_probs)
    |> Enum.map(fn {ck, px} -> calc_input_probs(input, ck, px, model.alpha) end)
    |> Nx.tensor()
    |> Nx.argmax()
    |> Nx.new_axis(0)
  end

  defp calc_input_probs(inputs, ck, px, alpha) do
    num_f = inputs[0] |> Nx.size()
    for f <- 0..(num_f - 1), reduce: ck do
      p -> p * Map.get(px, {f, Nx.to_scalar(inputs[0][f])}, alpha)
    end
  end

  defp calc_feature_probs(c, inputs, targets) do
    t_inputs = inputs
    |> Enum.zip(targets)
    |> Enum.filter(fn {_input, target} -> target |> Nx.squeeze() |> Nx.to_scalar() == c end)
    |> Enum.unzip()
    |> elem(0)
    |> Nx.concatenate()

    num_f = t_inputs[0] |> Nx.size()
    for f <- 0..(num_f - 1), reduce: %{} do
      acc -> f_vals = t_inputs[[0..-1, f]]
        vals = f_vals |> Nx.to_flat_list() |> MapSet.new() |> MapSet.to_list()
        px = f_vals |> frequencies() |> Nx.add(1) |> prob_dist() |> Nx.to_flat_list()
        vals
        |> Enum.zip(px)
        |> Enum.reduce(acc, fn {val, p}, a -> Map.put(a, {f, val}, p) end)
    end
  end

end
