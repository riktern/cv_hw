import statistics
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def prepare_data() -> TensorDataset:
    X = torch.randn(10000, 128)
    y = torch.randint(0, 2, (10000,))
    dataset = TensorDataset(X, y)
    return dataset


def train():
    #  Добавили pin_memory=True для ускорения пересылки CPU -> GPUи num_workers
    
    dataloader = DataLoader(prepare_data(), batch_size=256, shuffle=True, pin_memory=True)

    
    model = nn.Sequential(
        nn.Linear(128, 512), nn.ReLU(),
        nn.Linear(512, 128), nn.ReLU(),
        nn.Linear(128, 2)
    ).to('cuda').train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    losses_history = []
    forward_times = []
    backward_times = []

    #  Создаем CUDA-события для точного замера времени на GPU
    start_fwd = torch.cuda.Event(enable_timing=True)
    end_fwd = torch.cuda.Event(enable_timing=True)
    start_bwd = torch.cuda.Event(enable_timing=True)
    end_bwd = torch.cuda.Event(enable_timing=True)

    # Для асинхронной передачи данных на GPU
    device = torch.device('cuda')

    for batch_idx, (data, target) in enumerate(dataloader):
        # Переносим данные асинхронно 
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        #Генерируем шум на GPU
        noise = torch.randn(data.shape, device=device)
        data = data + noise

        optimizer.zero_grad()

        #Правильный замер Forward
        start_fwd.record()
        output = model(data)
        loss = criterion(output, target)
        end_fwd.record()

        #Правильный замер Backward
        start_bwd.record()
        loss.backward()
        end_bwd.record()
        
        optimizer.step()

        #Синхронизируем поток
        
        
        forward_times.append(start_fwd.elapsed_time(end_fwd) / 1000.0)
        backward_times.append(start_bwd.elapsed_time(end_bwd) / 1000.0)

        #Извлекаем только числовое значение 
        # Граф удаляется из памятb
        losses_history.append(loss.item()) 
        
        print(f"Batch {batch_idx} loss: {losses_history[-1]:.4f}")
        
        #Убрал torch.cuda.empty_cache() тк он ломал производительность
        # Теперь память очищается автоматически

    print(f"Epoch finished, avg forward time is {statistics.mean(forward_times):.6f}s, "
          f"avg backward time is {statistics.mean(backward_times):.6f}s")


if __name__ == '__main__':
    train()