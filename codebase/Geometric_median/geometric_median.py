import torch
from utils 									  import check_list_of_list_of_array_format, check_list_of_array_format
from Geometric_median.weiszfeld_array         import geometric_median_array, geometric_median_per_component
from Geometric_median.weiszfeld_list_of_array import geometric_median_list_of_array


def compute_geometric_median_ours(points, weights=None, per_component=True, skip_typechecks=False,
								  eps=1e-7, maxiter=100, ftol=1e-20):
	n_dim = points.dim()
	if type(points) == torch.Tensor:
		points = [p for p in points]
	else:
		raise ValueError("We expect points to be of type torch.Tensor .")

	if n_dim == 2:
		if not skip_typechecks:
			check_list_of_array_format(points)
		if weights is None:
			weights = torch.ones(len(points), device=points[0].device)
		to_return = geometric_median_array(points, weights, eps, maxiter, ftol)
	elif n_dim == 3:
		if not skip_typechecks:
			check_list_of_list_of_array_format(points)

		if per_component:
			if weights is None:
				weights = torch.ones(len(points[0]), device=points[0][0].device)
			to_return = geometric_median_per_component(points, weights, eps, maxiter, ftol)

		else:
			if weights is None:
				weights = torch.ones(len(points), device=points[0][0].device)
			to_return = geometric_median_list_of_array(points, weights, eps, maxiter, ftol)
	else:
		raise ValueError("We want Tensors to be in 2D or 3D.")
	return to_return.median

