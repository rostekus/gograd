package gograd

import (
	"math"
	"math/rand"
	"time"
)

func GenerateRandomFloat(min float64, max float64) float64 {
	rand.Seed(time.Now().UnixNano())

	upper := math.Ceil(min)
	lower := math.Floor(max)
	n := rand.Float64()*(upper-lower) + lower

	return n
}

type Neuron struct {
	Weights []*Value
	Bias    *Value
}

func NewNeuron(nin int) *Neuron {
	var weights []*Value
	for i := 0; i < nin; i++ {
		weights = append(weights, NewValue(GenerateRandomFloat(-1, 1)))
	}
	bias := NewValue(GenerateRandomFloat(-1, 1))

	neuron := &Neuron{
		Weights: weights,
		Bias:    bias,
	}

	return neuron
}

func (n Neuron) Call(x []*Value) *Value {
	act := n.Bias

	for i, _ := range x {
		act = act.Add(n.Weights[i].Mul(x[i]))
	}

	out := act.Tanh()
	return out
}

func (n Neuron) Parameters() []*Value {
	return append(n.Weights, n.Bias)
}

type Layer struct {
	Neurons []*Neuron
}

func NewLayer(nin int, nout int) *Layer {
	var neurons []*Neuron

	for i := 0; i < nout; i++ {
		neurons = append(neurons, NewNeuron(nin))
	}

	layer := &Layer{
		Neurons: neurons,
	}

	return layer
}

func (l Layer) Call(x []*Value) []*Value {
	var outs []*Value

	for _, n := range l.Neurons {
		outs = append(outs, n.Call(x))
	}

	return outs
}

func (l Layer) Parameters() []*Value {
	var params []*Value

	for _, neuron := range l.Neurons {
		params = append(params, neuron.Parameters()...)
	}

	return params
}

type MLP struct {
	Layers []*Layer
}

func NewMLP(nin int, nouts []int) *MLP {
	sz := append([]int{nin}, nouts...)

	var layers []*Layer

	for i, _ := range nouts {
		layers = append(layers, NewLayer(sz[i], sz[i+1]))
	}

	mlp := &MLP{
		Layers: layers,
	}

	return mlp
}

func (mlp MLP) Call(x []float64) *Value {
	var l []*Value

	for _, el := range x {
		l = append(l, NewValue(el))
	}

	for _, layer := range mlp.Layers {
		l = layer.Call(l)
	}

	return l[0]
}

func (mlp MLP) Parameters() []*Value {
	var params []*Value

	for _, layer := range mlp.Layers {
		params = append(params, layer.Parameters()...)
	}

	return params
}
