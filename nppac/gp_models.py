import torch
import gpytorch
from gpytorch.kernels import RBFKernel, MaternKernel, MultitaskKernel


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, rank=1, ard_num_dims=None,
                 kernel_type='matern12'):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        print('Using rank {} for the task covar matrix.'.format(rank))

        if kernel_type == 'matern12':
            self.base_kernel = MaternKernel(nu=0.5, ard_num_dims=ard_num_dims)
        elif kernel_type == 'matern32':
            self.base_kernel = MaternKernel(nu=1.5, ard_num_dims=ard_num_dims)
        elif kernel_type == 'matern52':
            self.base_kernel = MaternKernel(nu=2.5, ard_num_dims=ard_num_dims)
        elif kernel_type == 'rbf':
            self.base_kernel = RBFKernel(ard_num_dims=ard_num_dims)
        else:
            raise SystemExit('Invalid kernel type.')

        print('Using kernel type: {}.'.format(kernel_type))

        self.covar_module = MultitaskKernel(
            self.base_kernel, num_tasks=num_tasks, rank=rank
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class VariMultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, num_tasks, ard_num_dims=None, kernel_type='matern12'):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_tasks])
        )

        variational_strategy = gpytorch.variational.MultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ), num_tasks=num_tasks
        )

        super().__init__(variational_strategy)

        if kernel_type == 'matern12':
            self.base_kernel = MaternKernel(nu=0.5, batch_shape=torch.Size([num_tasks]))
        elif kernel_type == 'matern32':
            self.base_kernel = MaternKernel(nu=1.5, batch_shape=torch.Size([num_tasks]))
        elif kernel_type == 'matern52':
            self.base_kernel = MaternKernel(nu=2.5, batch_shape=torch.Size([num_tasks]))
        elif kernel_type == 'rbf':
            self.base_kernel = RBFKernel(batch_shape=torch.Size([num_tasks]))
        else:
            raise SystemExit('Invalid kernel type.')

        print('Using variational GP with kernel type: {}.'.format(kernel_type))

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            self.base_kernel, batch_shape=torch.Size([num_tasks])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, ard_num_dims=None,
                 kernel_type='matern12'):
        super(BatchIndependentMultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))

        if kernel_type == 'matern12':
            self.base_kernel = MaternKernel(nu=0.5, batch_shape=torch.Size([num_tasks]))
        elif kernel_type == 'matern32':
            self.base_kernel = MaternKernel(nu=1.5, batch_shape=torch.Size([num_tasks]))
        elif kernel_type == 'matern52':
            self.base_kernel = MaternKernel(nu=2.5, batch_shape=torch.Size([num_tasks]))
        elif kernel_type == 'rbf':
            self.base_kernel = RBFKernel(batch_shape=torch.Size([num_tasks]))
        else:
            raise SystemExit('Invalid kernel type.')

        print('Using batch independent GP with kernel type: {}.'.format(kernel_type))

        self.covar_module = gpytorch.kernels.ScaleKernel(
            self.base_kernel, batch_shape=torch.Size([num_tasks])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )
