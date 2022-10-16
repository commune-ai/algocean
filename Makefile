down:
	./start.sh --light
stop:

start:
	./start.sh --light --up
up:
	make start
restart:
	make down; make up;

backend: 
	./start.sh --backend; docker exec -it ocean_backend_1;

restart:
	./start.sh --purge && ./start.sh;


prune_volumes:	
	docker system prune --all --volumes

bash:
	docker exec -it ocean_${arg}_1 bash
build_backend:
	docker-compose -f "backend/backend.yml" build;

app:
	make streamlit
kill_all:
	docker kill $(docker ps -q)

logs:
	docker logs ocean_${arg}_1 --tail=100 --follow

streamlit:
	docker exec -it ocean_backend_1 bash -c "streamlit run algocean/${arg}/module.py "
enter_backend:
	docker exec -it ocean_backend_1 bash
pull:
	git submodule update --init --recursive
	
kill_all:
	docker kill $(docker ps -q) 

jupyter:
	docker exec -it ocean_backend_1 bash -c "jupyter lab --allow-root --ip=0.0.0.0 --port=8888"

python:
	docker exec -it ocean_backend_1 bash -c "python algocean/${arg}/module.py"

exec:
	docker exec -it ocean_backend_1 bash -c "${arg}"

