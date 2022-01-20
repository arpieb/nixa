defmodule Nixa.Dummy.RandomClassifier do
  @moduledoc """
  Implements a "dumb" classifier that randomly selects one of the classes, no fitting occurs.
  """

  defstruct [
    num_classes: nil
  ]

  @doc """
  No training, simply collect the max target class for use in randomization
  """
  def fit(_inputs, targets, _opts \\ []) do
    %__MODULE__{
      num_classes: targets |> Nx.concatenate() |> Nx.reduce_max() |> Nx.to_number(),
    }
  end

  @doc """
  Return a random class value for each input
  """
  def predict(%__MODULE__{} = model, inputs) do
    num_targets = Enum.count(inputs)
    Nx.random_uniform({num_targets}, 0, model.num_classes)
    |> Nx.to_batched_list(1)
  end

end
