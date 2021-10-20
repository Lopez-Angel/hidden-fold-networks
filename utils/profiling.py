# Copyright 2021 Angel Lopez Garcia-Arias

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        https://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

def estimate_params_size(model, args):

    total_batchnorm_bias = 0
    total_batchnorm_weights = 0
    total_dense_weights = 0
    total_sparse_weights = 0
    total_params = 0
    
    print(f"\nPARAMETER COUNT:")

    for n, m in model.named_modules():
        batchnorm_bias = 0
        batchnorm_weights = 0
        dense_weights = 0
        sparse_weights = 0
        layer_total = 0

        if hasattr(m, "bias") and m.bias is not None:
            # assuming conv layers without bias, this is a batchnorm layer
            batchnorm_bias = m.bias.numel()
            if hasattr(m, "weight") and m.weight is not None:
                batchnorm_weights = m.weight.numel()

        elif hasattr(m, "weight") and m.weight is not None:
            # assuming conv layers without bias, this is a conv layer
            dense_weights = m.weight.numel()
            sparse_weights = int(dense_weights * (args.top_k)) 

        layer_total = batchnorm_bias + batchnorm_weights + sparse_weights
        if(layer_total) > 0 :
            # print(f"\tLAYER {n}")
            # print(f"\t\t#BatchNorm bias: {batchnorm_bias}")
            # print(f"\t\t#BatchNorm weights: {batchnorm_weights}")
            # print(f"\t\t#Dense Conv weights: {dense_weights}")
            # print(f"\t\t#Used Conv weights: {sparse_weights}")
            # print(f"\t\tTotal #param: {layer_total}")

            total_batchnorm_bias += batchnorm_bias
            total_batchnorm_weights += batchnorm_weights
            total_dense_weights += dense_weights
            total_sparse_weights += sparse_weights

            total_params += layer_total

    print(f"\tTOTAL")
    print(f"\t\t#BatchNorm bias: {total_batchnorm_bias}")
    print(f"\t\t#BatchNorm weights: {total_batchnorm_weights}")
    print(f"\t\t#Dense Conv weights: {total_dense_weights}")
    print(f"\t\t#Used Conv weights: {total_sparse_weights}")
    print(f"\tThis model has an estimated total of {total_params/1000000} million parameters.")

    default_param_size = 32 #bits
    mask_size = total_dense_weights #bits
    rand_seed_size = default_param_size
    megabyte = 8000000

    vanilla_size = default_param_size * (total_batchnorm_bias + total_batchnorm_weights + total_dense_weights)
    vanilla_size = vanilla_size/ megabyte
    hidden_size = rand_seed_size + mask_size + default_param_size * (total_batchnorm_bias + total_batchnorm_weights)
    hidden_size = hidden_size/ megabyte

    print(f"Memory size if dense: {vanilla_size} MBytes")
    if args.top_k < 1.0:
        print(f"Memory size w/ supermask & compression: {hidden_size} MBytes")

    return (total_batchnorm_bias, total_batchnorm_weights, total_dense_weights, 
            total_sparse_weights, total_params, vanilla_size, hidden_size)
