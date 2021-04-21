defmodule Nixa.Stats do
  @moduledoc """
  Provides statistical functions
  """

  import Nx.Defn
  import Nixa.Shared

  defn entropy(px) do
    Nx.sum(-px * log2(px))
  end

  defn softmax(t) do
    Nx.divide(Nx.exp(t), Nx.sum(Nx.exp(t)))
  end

  defn prob_dist(t) do
    t / Nx.sum(t)
  end

end