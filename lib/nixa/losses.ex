defmodule Nixa.Losses do
  @moduledoc ~S"""
  Shamelessly borrowed from Axon.Losses (elixir-nx/axon)
  """

  import Nx.Defn
  import Nixa.Shared

  @doc ~S"""
  Categorical hinge loss function.
  ## Argument Shapes
    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$
  ## Examples
      iex> y_true = Nx.tensor([[1, 0, 0], [0, 0, 1]], type: {:s, 8})
      iex> y_pred = Nx.tensor([[0.05300799, 0.21617081, 0.68642382], [0.3754382 , 0.08494169, 0.13442067]])
      iex> Axon.Losses.categorical_hinge(y_true, y_pred)
      #Nx.Tensor<
        f32[2]
        [1.6334158182144165, 1.2410175800323486]
      >
  """
  defn categorical_hinge(y_true, y_pred) do
    1
    |> Nx.subtract(y_true)
    |> Nx.multiply(y_pred)
    |> Nx.reduce_max(axes: [-1])
    |> Nx.subtract(Nx.sum(Nx.multiply(y_true, y_pred), axes: [-1]))
    |> Nx.add(1)
    |> Nx.max(0)
  end

  @doc ~S"""
  Hinge loss function.
  $$\frac{1}{C}\max_i(1 - \hat{y_i} * y_i, 0)$$
  ## Argument Shapes
    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$
  ## Examples
      iex> y_true = Nx.tensor([[ 1,  1, -1], [ 1,  1, -1]], type: {:s, 8})
      iex> y_pred = Nx.tensor([[0.45440044, 0.31470688, 0.67920924], [0.24311459, 0.93466766, 0.10914676]])
      iex> Axon.Losses.hinge(y_true, y_pred)
      #Nx.Tensor<
        f32[2]
        [0.9700339436531067, 0.6437881588935852]
      >
  """
  defn hinge(y_true, y_pred) do
    assert_shape!(y_true, y_pred)

    y_true
    |> Nx.multiply(y_pred)
    |> Nx.negate()
    |> Nx.add(1)
    |> Nx.max(0)
    |> Nx.mean(axes: [-1])
  end
end
