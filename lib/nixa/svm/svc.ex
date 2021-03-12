defmodule Nixa.SVM.SVC do
  @moduledoc false

  import Nx.Defn
#  import Nixa.Shared
#  import Nixa.Losses

  @enforce_keys [:w, :lr]
  defstruct [:w, :lr]

  def create(num_features, opts \\ []) do
    w = Nx.broadcast(0.0, {num_features})
    lr = Keyword.get(opts, :learning_rate) || 1.0 |> Nx.tensor()

    %__MODULE__{w: w, lr: lr}
  end

  def fit(model, inputs, targets, opts \\ []) do
    epochs = Keyword.get(opts, :epochs) || 1
    {time, model} = :timer.tc(__MODULE__, :train, [model, inputs, targets, epochs])
    IO.puts("Total training time: #{time / 1_000_000}s")
    model
  end

  def train(model, inputs, targets, epochs) do
    {model, _, _, _} = for epoch <- 1..epochs, reduce: {model, inputs, targets, epochs} do
      {model, inputs, targets, epochs} ->
        #        {train_epoch(model, inputs, targets, epochs, epoch), inputs, targets, epochs}
        {time, model} = :timer.tc(__MODULE__, :train_epoch, [model, inputs, targets, epochs])
#        IO.puts("Epoch #{epoch} time: #{time / 1_000_000}s")
        {model, inputs, targets, epochs}
    end
    model
  end

  def train_epoch(model, inputs, targets, epochs) do
    {model, _} = inputs
    |> Enum.zip(targets)
    |> Enum.reduce({model, epochs}, fn {input, target}, {model, epochs} -> {update_model(model, input, target, epochs), epochs} end)

    model
  end

  defp update_model(model, input, target, epochs) do
    %{model | w: update_w(model.w, model.lr, input, target, epochs)}
  end

  defnp update_w(w, lr, x, y, epochs) do
    val1 = Nx.dot(x, w)
    if Nx.less(y * val1, 1.0) do
      w + lr * ((y * x) - (2.0 * (1.0 / epochs) * w))
    else
      w + lr * (-2.0 * (1.0 / epochs) * w)
    end
  end

end
