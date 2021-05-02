defmodule Nixa.Dummy.RandomRegressor do
  @moduledoc """
  Implements a "dumb" regressor that randomly a value between the min and max targets, no fitting occurs.
  """

  defstruct [
    min_target: nil,
    max_target: nil
  ]

  @doc """
  No training, simply collect the min/max target values for use in randomization
  """
  def fit(_inputs, targets, _opts \\ []) do
    t_targets = targets |> Nx.concatenate()
    %__MODULE__{
      min_target: t_targets |> Nx.reduce_min() |> Nx.to_scalar(),
      max_target: t_targets |> Nx.reduce_max() |> Nx.to_scalar(),
    }
  end

  @doc """
  Return a random value for each input between trained min/max values
  """
  def predict(%__MODULE__{} = model, inputs) do
    num_targets = Enum.count(inputs)
    Nx.random_uniform({num_targets}, model.min_target, model.max_target)
    |> Nx.to_batched_list(1)
  end

end
