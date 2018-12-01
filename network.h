#ifndef NETWORK_H
#define NETWORK_H

struct layer
{
    float *weights{};
    unsigned int n_inputs{};
    unsigned int n_units{};
};

struct network
{
    struct layer *layers{};
    unsigned int n_layers{};
};

#endif /* !NETWORK_H */
