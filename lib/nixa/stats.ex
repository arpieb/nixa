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

  @doc """
  Calculate the standard deviation of the provided tensor
  """
  def std(t, opts \\ []) do
    axes = Keyword.get(opts, :axes, nil)
    n = Nx.shape(t) |> elem(0)
    if axes == nil,
      do: Nx.mean(t) |> Nx.subtract(t) |> Nx.power(2) |> Nx.sum() |> Nx.divide(n) |> Nx.sqrt(),
      else: Nx.mean(t, axes: axes) |> Nx.subtract(t) |> Nx.power(2) |> Nx.sum(axes: axes) |> Nx.divide(n) |> Nx.sqrt()
  end

end
