defmodule Nixa.NaiveBayes.Multinomial do
  @moduledoc """
  Implements the Multinomial Naive Bayes algorithm
  """

  import Nx.Defn
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
    |> Enum.map(fn c -> Task.async(fn -> calc_feature_probs(c, inputs, targets, alpha) end) end)
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
    |> Enum.map(fn {ck, px} -> calc_input_probs(input, ck, px) end)
    |> Nx.stack()
    |> Nx.argmax()
    |> Nx.new_axis(0)
  end

  defnp calc_input_probs(input, ck, px) do
    Nx.select(Nx.greater(input, 0), px, 1.0)
    |> Nx.multiply(ck)
    |> Nx.product()
  end

  defp calc_feature_probs(c, inputs, targets, alpha) do
    t_inputs = inputs
    |> Enum.zip(targets)
    |> Enum.filter(fn {_input, target} -> target |> Nx.squeeze() |> Nx.to_scalar() == c end)
    |> Enum.unzip()
    |> elem(0)
    |> Nx.concatenate()

    calc_feature_probs_n(t_inputs, alpha)
  end

  defnp calc_feature_probs_n(t_inputs, alpha) do
    num_f = t_inputs[0] |> Nx.size()
    cf_count = t_inputs |> Nx.sum()
    f_counts = Nx.sum(t_inputs, axes: [0])
    (f_counts + alpha) / (cf_count + alpha * num_f)
  end

end
