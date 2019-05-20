module learndigit

using LinearAlgebra
using IDX

function sigmoid(x)
	return 1/(1+exp(-x))
end

function network(input,w,b)
	input = input[:]; #shape of input is (nin,)

	nhid = 15; #number of hidden layer nodes
	nout = 10; #number of output layer nodes

	w1,w2,b1,b2 = reshape_stuff(w,b,length(input),nhid,nout);	

	midlayer = sigmoid.(dot(input,w1)+b1); 
	out =  sigmoid.(dot(midlayer,w2)+b2);

	return dot((out .==maximum(out)	),[0:9]),out
end

function reshape_stuff(w,b,nin,nhid,nout)
	w1 = reshape(w[1:(nin*nhid)],(nhid,nin));
	w2 = reshape(w[(nin*nhid+1):end],(nout,nhid));
	b1 = b[1:(nhid)];
	b2 = b[(nhid+1):end];
	
	return w1,w2,b1,b2
end

function loadinputs()
	train_dat = load("data/train-images-idx3-ubyte");
	train_dat = train_dat[3];
	labs = load("data/train-labels-idx1-ubyte");
	labs = labs[3];
	return train_dat,labs
end

function training(eps)
	nhid = 15;
	nout = 10;

	dat,labs = loadinputs()
	while 

end

function gradc(dat,labs,w,b)

end

function cost(dat,labs,w,b)
	c = 0;
	nout = 10;
	subdat,sublabs = stoch_samp(dat,labs,0.1);
	for i = 1:size(subdat,3)
		guess = network(subdat[:,:,i],w,b);
		c = c+(guess-sublabs[i])^2;
	end

	return sqrt(c/length(labs))
end

function stoch_samp(a,b,frac)
	

end



end
