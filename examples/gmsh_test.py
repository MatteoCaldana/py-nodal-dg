import pyndg.mesh.generate as gen


if __name__ == "__main__":
    def callback(gmsh):
        node_tags, coords, parametric_coords = gmsh.model.mesh.getNodes()
        element_types, element_tags, node_tags = gmsh.model.mesh.getElements()

        print("Element Types:", element_types)
        print("Element Tags:", element_tags)
        print("Node Tags:", node_tags)  

        for et in element_types:
            name, _, _, _, _, _ = gmsh.model.mesh.getElementProperties(et)
            print(f"Type {et} is a '{name}'")


    gen.foo(callback=callback)