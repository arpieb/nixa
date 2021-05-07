defmodule Nixa.NaiveBayes.Shared do
  @moduledoc """
  Shared util functions for Naive Bayes implementations
  """

  import Nixa.Shared
  import Nixa.Stats

  def calc_class_prob(targets, :weighted, alpha) do
    targets |> Nx.concatenate() |> frequencies() |> Nx.add(alpha) |> prob_dist()
  end

  def calc_class_prob(targets, :equal, _alpha) do
    num_classes = targets |> Nx.concatenate() |> Nx.reduce_max() |> Nx.to_scalar()
    Nx.broadcast(1.0 / num_classes, {num_classes})
  end

end
