defmodule Nixa.Stats do
  @moduledoc """
  Provides statistical functions
  """

  import Nx.Defn
  import Nixa.Shared

  @doc """
  Calculate the entropy for a probability distribution
  """
  defn entropy(px) do
    Nx.sum(-px * log2(px))
  end

  @doc """
  Calculate the Gini impurity for a probability distribution
  """
  defn gini_impurity(px) do
    1.0 - Nx.sum(Nx.power(px, 2))
  end

  @doc """
  Calculate the softmax for  the provided tensor
  """
  defn softmax(t) do
    Nx.divide(Nx.exp(t), Nx.sum(Nx.exp(t)))
  end

  @doc """
  Calculate the probability distribution for a tensor
  """
  defn prob_dist(t) do
    t / Nx.sum(t)
  end

end
