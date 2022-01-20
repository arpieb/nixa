defmodule Nixa.NaiveBayes.Gaussian do
  @moduledoc """
  Implements the Gaussian Naive Bayes algorithm for continuous feature domains
  """

  import Nixa.NaiveBayes.Shared
  import Nixa.Stats
  import Nx.Defn

  defstruct [
    class_probs: nil,
    means: nil,
    stds: nil
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
    num_classes = class_probs |> Nx.size() |> Nx.to_number()
    {means, stds} = 0..(num_classes - 1)
      |> Enum.map(fn c -> Task.async(fn -> calc_feature_probs(c, inputs, targets) end) end)
      |> Task.await_many(:infinity)
      |> Enum.unzip()

    %__MODULE__{
      class_probs: class_probs,
      means: means,
      stds: stds,
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
    |> Enum.zip(model.means)
    |> Enum.zip(model.stds)
    |> Enum.map(fn {{ck, means}, stds} -> calc_input_probs(input, ck, means, stds, :math.pi) end)
    |> Nx.stack()
    |> Nx.argmax()
    |> Nx.new_axis(0)
  end

  defp calc_feature_probs(c, inputs, targets) do
    t_inputs = inputs
    |> Enum.zip(targets)
    |> Enum.filter(fn {_input, target} -> target |> Nx.squeeze() |> Nx.to_number() == c end)
    |> Enum.unzip()
    |> elem(0)
    |> Nx.concatenate()

    means = Nx.mean(t_inputs, axes: [0])
    stds = std(t_inputs, axes: [0])
    {means, stds}
  end

  defnp calc_input_probs(x, ck, means, stds, pi) do
    (1.0 / Nx.sqrt(2 * pi * Nx.power(stds, 2)) * Nx.exp(-0.5 * (((x - means) / stds) |> Nx.power(2))) |> Nx.product(axes: [0])) * ck
  end

end
