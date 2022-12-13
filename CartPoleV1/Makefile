
.PHONY: help
help:
	@echo "Usage: make <target>"
	@echo
	@echo "Main targets:"
	@echo
	@echo "-start_container     start the container"
	@echo "-build_container_img	build image for the container"
	@echo "-rm_img				remove the image"
	@echo "attach_container		attach to the created containre"


.PHONY: build_container_img
build_container_img:
	docker build --tag cart_pole_v1 .


.PHONY: start_container
start_container:
	docker run --name cart_pole_v1 --rm -detach --volume results:/scripts/results cart_pole_v1

.PHONY: rm_img
rm_img:
	docker rmi cart_pole_v1

.PHONY: attach_container
attach_container:
	docker exec -it cart_pole_v1 bash