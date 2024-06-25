#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mpi.h>
#include <numbers>
#include <petscksp.h>
#include <petsclog.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscpctypes.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <sys/types.h>
#include <vector>

#include <basix/finite-element.h>

#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/io/VTKFile.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/mesh/utils.h>

#include "poisson.h"

using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_type_t<T>;

int main(int argc, char** argv)
{
    int n_coarse = 8;
    int n_fine = 16;
    
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    // PetscLogDefaultBegin();

    auto mesh = std::make_shared<mesh::Mesh<U>>(
        dolfinx::mesh::create_interval<U>(MPI_COMM_SELF, n_fine, {0.0, 1.0}));

    auto element = basix::create_element<U>(
        basix::element::family::P, basix::cell::type::interval, 1,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);

    auto V = std::make_shared<fem::FunctionSpace<U>>(
        fem::create_functionspace<U>(mesh, element, {}));

    // Prepare and set Constants for the bilinear form
    auto f = std::make_shared<fem::Function<T>>(V);
    f->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
            f.push_back(-2 * std::numbers::pi * std::numbers::pi);
          return {f, {f.size()}};
        });

    {
        io::VTKFile file(MPI_COMM_SELF, "f.pvd", "w");
        file.write<T>({*f}, 0.0);
    }

    // Define variational forms
    auto a = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_poisson_a, {V, V}, {}, {}, {}));
    auto L = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_poisson_L, {V}, {{"f", f}}, {}, {}));
    
    auto A = la::petsc::Matrix(fem::petsc::create_matrix(*a), false);
    la::Vector<T> b(L->function_spaces()[0]->dofmap()->index_map,
                    L->function_spaces()[0]->dofmap()->index_map_bs());
    

    auto&& facets = mesh::locate_entities_boundary(*mesh, 0, [](auto x) { return std::vector<std::int8_t>(x.extent(1), true); });
    const auto bdofs = fem::locate_dofs_topological(*V->mesh()->topology_mutable(), *V->dofmap(), 0, facets);
    auto bc = std::make_shared<const fem::DirichletBC<T>>(0.0, bdofs, V);

    MatZeroEntries(A.mat());
    fem::assemble_matrix(la::petsc::Matrix::set_block_fn(A.mat(), ADD_VALUES), *a, {bc});
    MatAssemblyBegin(A.mat(), MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FLUSH_ASSEMBLY);
    fem::set_diagonal<T>(la::petsc::Matrix::set_fn(A.mat(), INSERT_VALUES), *V, {bc});
    MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);

    b.set(0.0);
    fem::assemble_vector(b.mutable_array(), *L);
    fem::apply_lifting<T, U>(b.mutable_array(), {a}, {{bc}}, {}, T(1));
    b.scatter_rev(std::plus<T>());
    fem::set_bc<T, U>(b.mutable_array(), {bc});

    KSP ksp;
    KSPCreate(MPI_COMM_WORLD, &ksp);
    KSPSetType(ksp, "preonly");

    PC pc;
    KSPGetPC(ksp, &pc);
    KSPSetFromOptions(ksp);
    PCSetType(pc, "mg");

    PCMGSetLevels(pc, 2, NULL);
    PCMGSetType(pc, PC_MG_MULTIPLICATIVE);
    PCMGSetCycleType(pc, PC_MG_CYCLE_V);
    PCMGSetGalerkin(pc, PC_MG_GALERKIN_BOTH);
    PCMGSetNumberSmooth(pc, 2);
    PCSetFromOptions(pc);

    Mat interpolation;
    {
        int64_t nz = 2*n_fine + (n_fine+1);
        std::vector<PetscInt> i; // row indices
        i.reserve(nz);
        std::vector<PetscInt> j; // col indices
        j.reserve(nz);
        std::vector<PetscScalar> a;
        a.reserve(nz);
        for (int64_t idx = 0; idx < n_fine+1; idx ++)
        {
            if (idx % 2 == 0)
            {
                i.emplace_back(idx);
                j.emplace_back(PetscInt(idx/2.));
                a.emplace_back(1);
            } else {
                i.emplace_back(idx);
                j.emplace_back(floor(idx/2.));
                a.emplace_back(.5);            

                i.emplace_back(idx);
                j.emplace_back(ceil(idx/2.));
                a.emplace_back(.5);
            }
        }
        MatCreateSeqAIJFromTriple(MPI_COMM_SELF, n_fine+1, n_coarse+1, i.data(), j.data(), a.data(), &interpolation, a.size(), PETSC_FALSE);
        MatView(interpolation, PETSC_VIEWER_STDOUT_SELF);
    }

    Mat restriction;
    MatTranspose(interpolation, MAT_INITIAL_MATRIX, &restriction);
    MatView(restriction, PETSC_VIEWER_STDOUT_SELF);

    PCMGSetInterpolation(pc, 1, interpolation);
    PCMGSetRestriction(pc, 1, restriction);

    MatView(A.mat(), PETSC_VIEWER_STDOUT_SELF);

    KSPSetOperators(ksp, A.mat(), A.mat());
    KSPSetUp(ksp);

    auto u = std::make_shared<fem::Function<T>>(V);

    la::petsc::Vector _u(la::petsc::create_vector_wrap(*u->x()), false);
    la::petsc::Vector _b(la::petsc::create_vector_wrap(b), false);

    // VecView(_b.vec(), PETSC_VIEWER_STDOUT_SELF);

    KSPSolve(ksp, _b.vec(), _u.vec());
    // Update ghost values before output
    u->x()->scatter_fwd();
    // VecView(_u.vec(), PETSC_VIEWER_STDOUT_SELF);

    {
        io::VTKFile file(MPI_COMM_SELF, "u.pvd", "w");
        file.write<T>({*u}, 0.0);
    }
    // PCView(pc, PETSC_VIEWER_STDERR_SELF);
    
    KSPDestroy(&ksp);
}