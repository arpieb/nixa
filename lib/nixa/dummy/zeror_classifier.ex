defmodule Nixa.Dummy.ZeroRClassifier do
  @moduledoc """
  Implements the ZeroR classifier algorithm
  """

  import Nixa.Shared

  defstruct [
    best_target: nil
  ]

  @doc """
  No training, simply collect the max target class for use in randomization
  """
  def fit(_inputs, targets, _opts \\ []) do
    %__MODULE__{
      best_target: get_argmax_target(targets) |> Nx.squeeze() |> Nx.to_scalar()
    }
  end

  @doc """
  Return a random class value for each input
  """
  def predict(%__MODULE__{} = model, inputs) do
    num_targets = Enum.count(inputs)
    Nx.broadcast(model.best_target, {num_targets})
    |> Nx.to_batched_list(1)
  end

end
