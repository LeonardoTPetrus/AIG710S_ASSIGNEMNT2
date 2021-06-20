### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 1598cc4c-ce91-11eb-22be-e95b0eee4d58
using FastAI

# ╔═╡ a005a1f9-5b34-471f-816d-4f4821490256
begin
	using FastAI.Datasets
	using FastAI.Datasets: datasetpath, loadtaskdata
	using FastAI.Datasets: FileDataset
	using FastAI: DLPipelines
end

# ╔═╡ 97996282-dbbd-4b11-909d-996b4693085c
using FastAI.Datasets: loadfile, filename

# ╔═╡ 7221e043-a0e6-48d1-9fdf-3e935110bb7f
begin
	using Flux
	using Flux: onehotbatch, onecold, crossentropy, throttle
	using Base.Iterators: repeated, partition
	using Printf, BSON
	using ImageView
	using ImageIO
	using Plots
end

# ╔═╡ cc0cee1a-b41c-4b06-a4e3-c3c144e609b6
using PlutoUI, DecisionTree, QuartzImageIO

# ╔═╡ 81228e80-421c-4bd5-8588-d4cb59fc9f79
using Statistics

# ╔═╡ 7d4aa1ac-884a-44e4-937b-9b1d156e009c
using Flux: @epochs

# ╔═╡ 14cc85aa-e533-46ad-9593-f379517e6852
using LinearAlgebra: QR

# ╔═╡ 7706c0da-e5f9-4950-a492-4bf14b2daee3
begin
	using DataAugmentation
	using Colors: RGB
	using FastAI: IMAGENET_MEANS, IMAGENET_STDS
end

# ╔═╡ 163eef39-d137-4022-bc87-9e439193f9cf
md"# AIG710S ASSIGNMENT 2"

# ╔═╡ 5e4aceec-29e6-475e-b071-191889b57b86
md"### Packages
> Importing The different packages needed"

# ╔═╡ 265133e3-4201-4e90-ad2a-c10b3210d3ef
md"## ------------------------------------------------"

# ╔═╡ 04ca85bd-456d-4297-afb1-c2cdec72557d
md"### Image Processing using FastAI"

# ╔═╡ 0b974aa8-f8d6-4a7d-af13-a037292c814c
md"> Creating data containers from files using the FastAI package. The data containers load the images in a specific path and return the number of observations"

# ╔═╡ 366b0dcb-8d56-4c10-b215-a35505d49300
filedata = FileDataset("/Users/leonardopetrus/Desktop/CHEST_X-RAY")

# ╔═╡ 72d87139-ad8e-42d0-b5f4-cb940901e9d6
filedataTest = FileDataset("/Users/leonardopetrus/Desktop/AIG/chest_xray/chest_xray/test")

# ╔═╡ e882fc6e-de23-4e13-935a-5991b81ad1a7
filedataTrain = FileDataset("/Users/leonardopetrus/Desktop/AIG/chest_xray/chest_xray/train")

# ╔═╡ 2305317a-f08d-45e7-bc2c-e0373ae5eebc
filedataVal = FileDataset("/Users/leonardopetrus/Desktop/AIG/chest_xray/chest_xray/val")

# ╔═╡ da4f5944-0e22-4848-a3b9-0f084172cd60
filedataTrainN = FileDataset("/Users/leonardopetrus/Desktop/CHEST_X-RAY/train/NORMAL")

# ╔═╡ 966b3e5b-85a8-4c38-a712-bd30bcf359ca
filedataTrainP = FileDataset("/Users/leonardopetrus/Desktop/CHEST_X-RAY/train/PNEUMONIA")

# ╔═╡ 16bb4d56-de2d-473b-a24a-38fd17b93750
#This line of code basically allows me to confirm wether there are images in the path of the data container that can be loaded 
p = getobs(filedata, 600)

# ╔═╡ 08796e3c-d653-4633-831c-8169f48217c9
#The following function is to load an image and class from the given path in a specific data container
begin
	function loadimageclass(p)
	    return (
	        Datasets.loadfile(p),
	        filename(parent(p)),
	    )
	end
	
	image, class = loadimageclass(p)
	@show class
	image
end

# ╔═╡ 6d2ef4df-78f1-4a61-b138-52ba5493a5a4
md"> The following functions are used to resize the images to 32x32 images in the data containers and change the colour of the images to perform easier teaching of the model"

# ╔═╡ 36db6774-2600-43fa-b797-3f5d6eb0991c
function resize_and_grayify(filedata, im_name, width=32, height=32)
    resized_gray_img = Gray.(load(filedata * "/" * im_name)) |> (x -> imresize(x, width, height))
    try
        save("preprocessed_" * filedata * "/" * im_name, resized_gray_img)
    catch e
        if isa(e, SystemError)
            mkdir("preprocessed_" * filedata)
            save("preprocessed_" * filedata * "/" * im_name, resized_gray_img)
        end
    end
end

# ╔═╡ d91fc110-03bf-4475-ba72-63b9faf54229
function process_images(filedata, width=32, height=32)
    files_list = readdir(filedata)
    map(x -> resize_and_grayify(filedata, x, width, height), files_list)
end

# ╔═╡ 15a3b943-e21a-4ca5-b37b-f68cf8a2189d
md"* Once images have been loaded, we concatenate the normal and pneumonia images in the train set"

# ╔═╡ 820c67c2-97ca-4339-956d-aee89c8a52d8
normal = vcat(filedataTrainN)

# ╔═╡ 7a1e95dd-f959-4d50-b4b3-bf9a7ccf912a
pneumonia = vcat(filedataTrainP)

# ╔═╡ 6c3c3c80-3c09-4850-9bf7-aad93526e48e
data = vcat(normal,pneumonia)
#data = mapobs(loadimageclass, filedata);

# ╔═╡ a6ba9e73-452d-4406-aa9b-c5504531865c
#Labeling images. 0 for normal and 1 for pneumonia and the split the data and lables into a train and test set
begin
    labels = vcat([0 for _ in 1:length(normal)], [1 for _ in 1:length(pneumonia)])
    (x_train, y_train), (x_test, y_test) = splitobs(shuffleobs((data, labels)), at = 0.7)
end

# ╔═╡ 1064638d-fdb2-4e6b-91b1-797811103060
#Creating minibatches for training the network
function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    Y_batch = onehotbatch(Y[idxs], 0:1)
    return (X_batch, Y_batch)
end

# ╔═╡ 0d414133-5da9-4df5-9ef2-63d3ed07a4ac
begin
    # Defining the train and test sets.
    batchsize = 128
    mb_idxs = partition(1:length(x_train), batchsize)
    train_set = [make_minibatch(x_train, y_train, i) for i in mb_idxs]
    test_set = make_minibatch(x_test, y_test, 1:length(x_test));
end

# ╔═╡ 3f69f5b3-28b5-4096-8c15-eec66d1cc277
md"## CNN MODEL: LeNet5"

# ╔═╡ 072779cc-898e-47d7-a781-2bd2aee80da9
md">Using the LeNet5 architecture to build the model"

# ╔═╡ 0c2207dc-51f2-461a-a016-e699c94fbf10
begin
	function LeNet5(; imgsize=(28,28,1), nclasses=10) 
    out_conv_size = (imgsize[1]÷4 - 3, imgsize[2]÷4 - 3, 16)
    
    return Chain(
            x -> reshape(x, imgsize..., :),
            Conv((5, 5), imgsize[end]=>6, relu),
            MaxPool((2, 2)),
            Conv((5, 5), 6=>16, relu),
            MaxPool((2, 2)),
            x -> reshape(x, :, size(x, 4)),
            Dense(prod(out_conv_size), 120, relu), 
            Dense(120, 84, relu), 
            Dense(84, nclasses)
          )
	end
end

# ╔═╡ 720e812e-7727-497e-a1e4-bfa2fa139515
model = LeNet5

# ╔═╡ 7efe648a-b32d-41fd-81d1-8ec88406f1af
begin		
 	train_loss = Float64[]
    test_loss = Float64[]
    acc = Float64[]
    ps = Flux.params(model)
    opt = ADAM()
    L(x, y) = Flux.crossentropy(model(x), y)
    L((x,y)) = Flux.crossentropy(model(x), y)
    accuracy(x, y, f) = mean(Flux.onecold(f(x)) .== Flux.onecold(y))
    
    function update_loss!()
        push!(train_loss, mean(L.(train_set)))
        push!(test_loss, mean(L(test_set)))
        push!(acc, accuracy(test_set..., model))
        @printf("train loss = %.2f, test loss = %.2f, accuracy = %.2f\n", train_loss[end], test_loss[end], acc[end])
    end
end

# ╔═╡ 3254f9b0-2b54-4cf1-9715-5cd3467da84c
# here we train our model for n_epochs times.
@epochs 10 Flux.train!(L, ps, train_set, opt;
               cb = Flux.throttle(update_loss!, 8))

# ╔═╡ 5ec9e881-59ae-4b95-b093-02b4a6162803
begin
    plot(train_loss, xlabel="Iterations", title="Model Training", label="Train loss", lw=2, alpha=0.9)
    plot!(test_loss, label="Test loss", lw=2, alpha=0.9)
    plot!(acc, label="Accuracy", lw=2, alpha=0.9)
end

# ╔═╡ 5c2df2e8-a775-4b83-bbcd-99408ec660ae
opt_2 = ADAM(0.0005)

# ╔═╡ c3ac01f6-7bca-4846-950e-e4457dcb820c
@epochs 5 Flux.train!(L, ps, train_set, opt_2;
               cb = Flux.throttle(update_loss!, 8))

# ╔═╡ fad8fe65-6d82-4abb-8be6-0c6fd57dc089
begin
    plot(train_loss, xlabel="Iterations", title="Model Training", label="Train loss", lw=2, alpha=0.9, legend = :right)
    plot!(test_loss, label="Test loss", lw=2, alpha=0.9)
    plot!(acc, label="Accuracy", lw=2, alpha=0.9)
    vline!([82], lw=2, label=false)
end

# ╔═╡ Cell order:
# ╟─163eef39-d137-4022-bc87-9e439193f9cf
# ╟─5e4aceec-29e6-475e-b071-191889b57b86
# ╠═1598cc4c-ce91-11eb-22be-e95b0eee4d58
# ╠═a005a1f9-5b34-471f-816d-4f4821490256
# ╠═97996282-dbbd-4b11-909d-996b4693085c
# ╠═7221e043-a0e6-48d1-9fdf-3e935110bb7f
# ╠═cc0cee1a-b41c-4b06-a4e3-c3c144e609b6
# ╠═81228e80-421c-4bd5-8588-d4cb59fc9f79
# ╠═7d4aa1ac-884a-44e4-937b-9b1d156e009c
# ╠═14cc85aa-e533-46ad-9593-f379517e6852
# ╠═7706c0da-e5f9-4950-a492-4bf14b2daee3
# ╟─265133e3-4201-4e90-ad2a-c10b3210d3ef
# ╟─04ca85bd-456d-4297-afb1-c2cdec72557d
# ╟─0b974aa8-f8d6-4a7d-af13-a037292c814c
# ╠═366b0dcb-8d56-4c10-b215-a35505d49300
# ╠═72d87139-ad8e-42d0-b5f4-cb940901e9d6
# ╠═e882fc6e-de23-4e13-935a-5991b81ad1a7
# ╠═2305317a-f08d-45e7-bc2c-e0373ae5eebc
# ╠═da4f5944-0e22-4848-a3b9-0f084172cd60
# ╠═966b3e5b-85a8-4c38-a712-bd30bcf359ca
# ╠═16bb4d56-de2d-473b-a24a-38fd17b93750
# ╠═08796e3c-d653-4633-831c-8169f48217c9
# ╟─6d2ef4df-78f1-4a61-b138-52ba5493a5a4
# ╠═36db6774-2600-43fa-b797-3f5d6eb0991c
# ╠═d91fc110-03bf-4475-ba72-63b9faf54229
# ╟─15a3b943-e21a-4ca5-b37b-f68cf8a2189d
# ╠═820c67c2-97ca-4339-956d-aee89c8a52d8
# ╠═7a1e95dd-f959-4d50-b4b3-bf9a7ccf912a
# ╠═6c3c3c80-3c09-4850-9bf7-aad93526e48e
# ╠═a6ba9e73-452d-4406-aa9b-c5504531865c
# ╠═1064638d-fdb2-4e6b-91b1-797811103060
# ╠═0d414133-5da9-4df5-9ef2-63d3ed07a4ac
# ╟─3f69f5b3-28b5-4096-8c15-eec66d1cc277
# ╟─072779cc-898e-47d7-a781-2bd2aee80da9
# ╠═0c2207dc-51f2-461a-a016-e699c94fbf10
# ╠═720e812e-7727-497e-a1e4-bfa2fa139515
# ╠═7efe648a-b32d-41fd-81d1-8ec88406f1af
# ╠═3254f9b0-2b54-4cf1-9715-5cd3467da84c
# ╠═5ec9e881-59ae-4b95-b093-02b4a6162803
# ╠═5c2df2e8-a775-4b83-bbcd-99408ec660ae
# ╠═c3ac01f6-7bca-4846-950e-e4457dcb820c
# ╠═fad8fe65-6d82-4abb-8be6-0c6fd57dc089
