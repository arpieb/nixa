defmodule Nixa.Dummy.ZeroRRegressor do
  @moduledoc """
  Implements a kludgy attempt at a ZeroR regressor algorithm
  """

  defstruct [
    best_target: nil
  ]

  @doc """
  No training, simply collect the max target class for use in randomization
  """
  def fit(_inputs, targets, _opts \\ []) do
    %__MODULE__{
      best_target: targets |> Nx.concatenate() |> Nx.mean() |> Nx.to_scalar()
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
