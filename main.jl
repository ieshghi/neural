module learndigit

using LinearAlgebra
include("idx.jl")

function sigmoid(x)
	return 1/(1+exp(-x))
end
function dsigmoid(x)
	return exp(-x)/(1+exp(-x))^2
end

function init()
	nin = 28*28; #Size of the input, in this case a 28*28 image
	nhid = [15]; #number of hidden layer nodes for each layer
	nout = 10; #number of output layer nodes
	w = Matrix{Array{Float64}}(undef,length(nhid),1);
	b = Matrix{Array{Float64}}(undef,length(nhid),1);
	s = [nin;nhid;nout];
	for i = 1:length(nhid)
		j = i+1;
		w[i] = rand(s[j],s[i]);
		b[i] = rand(s[j]);
	end	
	return w,b
end

function network(input,w,b)
	input = input[:]; #shape of input is (nin,)
	out = 0;

	for i = 1:length(w)
		out = sigmoid.(w[i]*input+b[i]);
		input = out;		
	end

	return out
end

function loadinputs()
	train_dat = IDX.load("data/train-images-idx3-ubyte");
	train_dat = train_dat[3];
	labs = IDX.load("data/train-labels-idx1-ubyte");
	labs = labs[3];
	return train_dat,labs
end

function training(eps)
	dat,labs = loadinputs();
	w,b = init();
	frac = 0.1;
	eta = 0.01;
	epoch = 1;
	while cost(dat,labs,w,b,frac)>eps
		println("Epoch: ",epoch)
		w,b = update(dat,labs,w,b,eta,frac);
	end
	return w,b
end

function update(dat,labs,w,b,eta,frac)
	subdat,sublabs = stoch_samp(dat,labs,frac);
	gw = Matrix{Array{Float64}}(undef,length(w),1);
	gb = Matrix{Array{Float64}}(undef,length(w),1);

	for i = 1:size(subdat,3)
		dw,db = backprop(subdat[:,:,i],sublabs[i]);
		for j = 1:length(w)
			gw[j] = gw[j] + dw[j];
			gb[j] = gb[j] + db[j];
		end
	end
	
	for i = 1:length(w)
		w[i] = w[i] - eta.*gw[i]./length(sublabs);
		b[i] = b[i] - eta.*gb[i]./length(sublabs);
	end
	return w,b
end

function cost(dat,labs,w,b,frac)
	c = 0;
	subdat,sublabs = stoch_samp(dat,labs,frac);
	for i = 1:size(subdat,3)
		guess = network(subdat[:,:,i],w,b);
		c = c+(guess-sublabs[i])^2;
	end

	return c./length(sublabs)
end

function stoch_samp(a,b,frac)
	n = length(a[1,1,:]);
	if frac == 0
		ind = floor(rand()*n);
		aout = a[:,:,ind];
		bout = b[ind];
	else
		ind = rand(n).<frac;
		aout = a[:,:,ind];
		bout = b[ind];
	end
	return aout,bout
end

function backprop(x,y,w,b)
	nw = Matrix{Array{Float64}}(undef,length(w),1);
	nb = Matrix{Array{Float64}}(undef,length(w),1);
	for i = 1:length(w)
		nw[i] = zeros(size(w[i]));
		nb[i] = zeros(size(b[i]));
	end

	act = x;
	acthist = [x];
	zs = Matrix{Array{Float64}}(undef,length(w),1);
	for i = 1:length(w)
		z = w[i]*act+b;
		zs[i] = z;
		act = sigmoid.(z);
		acthist = [acthist,act];
	end
	
	delta = (acts[end]-y)*dsigmoid.(zs[end]);
	nb[end] = delta;
	nw[end] = delta*(acts[end-1]');

	for j = 1:length(w)
		z = zs[end-j];
		spz = dsigmoid.(z);
		delta = ((w[end-j+1]')*delta)*spz;
		nb[end-j] = delta;
		nw[end-j] = delta*acts[end-j-1];
	end
	return nw,nb
end

end
