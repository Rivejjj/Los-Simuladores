import simpy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Client:
	def __init__(self, env, resources, processing_rate):
		#tiempo procesado
		self.processing_rate = processing_rate

		#cosas de la simu
		self.env = env
		self.resources = resources

		#estadisticas
		self.t_queue_entry = env.now
		self.t_processor_entry = -2
		self.t_processor_exit = -3

	def move_on(self, processing_clients, clients_in_queue, proccesed_clients):
		processing_time = np.random.exponential(1 / self.processing_rate) #processing rate == lambda == 1/media == 1/scale
		clients_in_queue[0] += 1

		with self.resources.request() as req:
			yield req
			self.t_processor_entry = self.env.now
			clients_in_queue[0] -= 1
			processing_clients[0] += 1
			yield self.env.timeout(processing_time)
			#actualiza estadisticas particulares del problema
			self.t_processor_exit = self.env.now
			processing_clients[0] -= 1

		proccesed_clients.append(self)

class QueueSimulator:
	def __init__(self, arrival_rate, processing_rate, ammount_processors, system_size=float('inf')):
		self.ammount_processors = ammount_processors
		self.queue_length = system_size - ammount_processors
		self.time_to_simulate = None

		self.clients_in_queue_amount = [0]
		self.processing_clients_amount = [0]
		self.rejected_clients = 0
		self.accepted_clients = 0
		
		self.processed_clients = []
		self.clients_in_queue_samples = []
		self.processing_clients_samples = []

		self.env = None
		self.arrival_rate = arrival_rate
		self.processing_rate = processing_rate

	def client_generator(self, resources, arrival_rate, processing_rate):
		while True:
			arrival_time = np.random.exponential(1 / arrival_rate) #arrival rate == lambda == 1/media == 1/scale
			yield self.env.timeout(arrival_time)
			if (self.processing_clients_amount[0] < self.ammount_processors) or (self.clients_in_queue_amount[0] < self.queue_length):
				self.accepted_clients += 1
				client = Client(self.env, resources, processing_rate)
				self.env.process(client.move_on(self.processing_clients_amount, self.clients_in_queue_amount, self.processed_clients))
			else:
				self.rejected_clients += 1
	
	def look_at_length(self, t_length):
		while True:
			self.processing_clients_samples.append(self.processing_clients_amount[0])
			self.clients_in_queue_samples.append(self.clients_in_queue_amount[0])
			yield self.env.timeout(t_length)

	def run(self, time_to_simulate, stats_samples=10000):
		self.env = simpy.Environment()
		resources = simpy.Resource(self.env, self.ammount_processors)
		self.env.process(self.client_generator(resources, self.arrival_rate, self.processing_rate))
		self.env.process(self.look_at_length(time_to_simulate/stats_samples))
		self.env.run(until=time_to_simulate)
		
	def data_functions(self):
		return [
			'average_clients_in_queue',
			'average_clients_in_system',
			'average_time_in_queue'
			'average_time_in_system'
			'rejected_ratio',
		]

	def average_clients_in_queue(self):
		return sum(self.clients_in_queue_samples) / len(self.clients_in_queue_samples)
	
	def average_processing_ocupation(self): #cantidad de procesadores ocupados == #cantidad de usuarios trabajando
		return sum(self.processing_clients_samples) / len(self.processing_clients_samples)
	
	def average_clients_in_system(self):
		return self.average_clients_in_queue() + self.average_processing_ocupation()

	def average_time_in_queue(self):
		aux = 0
		for client in self.processed_clients:
			aux += client.t_processor_entry - client.t_queue_entry
		return aux / len(self.processed_clients)
	
	def average_time_in_system(self):
		aux = 0
		for client in self.processed_clients:
			aux += client.t_processor_exit - client.t_queue_entry
		return aux / len(self.processed_clients)

	def rejected_ratio(self):
		return self.rejected_clients/ (self.accepted_clients + self.rejected_clients)

	def frec_ocupated_proccesor(self, x):
		aux = 0
		for p in self.processing_clients_samples:
			if (p == x):
				aux += 1
		return aux/len(self.processing_clients_samples)
		
		#vect = [cantidad == x for cantidad in self.processing_clients_samples]
		#return sum(vect)/len(vect)	

	def plot_N_dist(self):
		dic = {}
		for i in range(len(self.clients_in_queue_samples)):
			used_proccesors = self.clients_in_queue_samples[i] + self.processing_clients_samples[i]
			dic[used_proccesors] = dic.get(used_proccesors, 0) + 1
		
		printeo = {}
		for i in dic.items():
			printeo[i[0]] = i[1]/len(self.clients_in_queue_samples)

		plt.bar(printeo.keys(), printeo.values())

		# Configurar etiquetas y título
		plt.xlabel('Elementos')
		plt.ylabel('Probabilidades')
		plt.title('Probabilidades de los elementos')
		plt.show()

def main():
	q = QueueSimulator(1, 0.01, float('inf'))
	print('alexa, play "musica de ascensor, 10 horas"')
	q.run(1e5, 1e5)
	print(f'rejected ratio: {q.rejected_ratio()}')
	print(f'avg clientes queue: {q.average_clients_in_queue()}')
	print(f'avg clientes system: {q.average_clients_in_system()}')
	print(f'avg time in queue: {q.average_time_in_queue()}')
	print(f'avg time in system: {q.average_time_in_system()}')
	print()
	i = 0
	print(f'frecuency of processing ocupation == {i}: {q.frec_ocupated_proccesor(i)}')
	print(f'avg processing ocupation: {q.average_processing_ocupation()}')

	q.plot_N_dist()
	


main()