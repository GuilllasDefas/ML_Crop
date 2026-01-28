import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
import tkinter as tk
from tkinter import filedialog, messagebox
import threading

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Predição em: {DEVICE}")

# ============ MODELO (IDÊNTICO AO TREINAMENTO) ============
class MarginAwareCropModel(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            backbone = models.efficientnet_b0(weights=None)
        except TypeError:
            backbone = models.efficientnet_b0(pretrained=False)
        
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.regressor = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1280, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 4)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        return torch.sigmoid(self.regressor(x))


# ============ FUNÇÃO DE CORTE COM PRECISÃO EXTREMA ============
def predict_crop(model, image_path, img_size=300):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Não foi possível carregar {image_path}")
    
    orig_h, orig_w = img.shape[:2]
    
    # Pré-processar
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    tensor = transforms.ToTensor()(img_resized)
    tensor = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )(tensor).unsqueeze(0).to(DEVICE)
    
    # Predição com mixed precision
    model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                pred = model(tensor)[0].cpu().numpy()
        else:
            pred = model(tensor)[0].cpu().numpy()
    
    # Desnormalizar com clamping rigoroso
    x1 = int(np.clip(np.round(pred[0] * orig_w), 0, orig_w - 10))
    y1 = int(np.clip(np.round(pred[1] * orig_h), 0, orig_h - 10))
    x2 = int(np.clip(np.round(pred[2] * orig_w), x1 + 10, orig_w))
    y2 = int(np.clip(np.round(pred[3] * orig_h), y1 + 10, orig_h))
    
    # Garantir margens mínimas de 3 pixels para evitar cortes colados
    min_margin = 5
    x1 = max(0, x1 - min_margin)
    y1 = max(0, y1 - min_margin)
    x2 = min(orig_w, x2 + min_margin)
    y2 = min(orig_h, y2 + min_margin)
    
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Predição inválida: coordenadas incorretas")
    
    # Aplicar corte
    cropped = img[y1:y2, x1:x2]
    return cropped, (x1, y1, x2 - x1, y2 - y1)  # (x, y, width, height)


# ============ INTERFACE TKINTER (SEMPRE EM PRIMEIRO PLANO) ============
class CropApp:
    def __init__(self, root):
        self.root = root
        self.root.title("✂️ Crop Inteligente - Precisão EXTREMA de Margens")
        self.root.geometry("550x320")
        self.root.resizable(False, False)
        
        # Forçar primeiro plano
        self.root.attributes('-topmost', True)
        self.root.update()
        self.root.attributes('-topmost', False)
        
        # Carregar modelo
        try:
            print("Carregando modelo...")
            checkpoint_path = "models/best_model.pth"
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Modelo não encontrado em: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            self.model = MarginAwareCropModel().to(DEVICE)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.img_size = checkpoint.get('img_size', 300)
            self.train_iou = checkpoint.get('iou', 0.0)
            self.train_margin_err = checkpoint.get('margin_error', 0.0)
            
            print(f"✅ Modelo carregado! (IoU: {self.train_iou:.4f} | Erro Margem: {self.train_margin_err:.6f})")
        except Exception as e:
            messagebox.showerror("❌ Erro Fatal", f"Não foi possível carregar o modelo:\n{str(e)}")
            root.quit()
            return
        
        # UI
        tk.Label(root, text="✂️ Crop Inteligente com Precisão EXTREMA de Margens", 
                font=("Arial", 16, "bold"), fg="#2c3e50").pack(pady=15)
        
        tk.Label(root, text=f"Modelo treinado: IoU={self.train_iou:.4f} | Erro Margem={self.train_margin_err:.6f}",
                font=("Arial", 9), fg="#7f8c8d").pack()
        
        tk.Label(root, text="\nSelecione o modo de processamento:", 
                font=("Arial", 12)).pack(pady=10)
        
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="📁 Processar Pasta Inteira", 
                 command=lambda: self.select_path(True),
                 width=30, height=2, bg="#27ae60", fg="white",
                 font=("Arial", 11, "bold"), relief=tk.RAISED, borderwidth=2).pack(pady=8)
        
        tk.Button(btn_frame, text="🖼️ Processar Uma Imagem", 
                 command=lambda: self.select_path(False),
                 width=30, height=2, bg="#3498db", fg="white",
                 font=("Arial", 11, "bold"), relief=tk.RAISED, borderwidth=2).pack(pady=8)
        
        tk.Label(root, text="⚠️ As imagens cortadas serão salvas na subpasta 'output' com sufixo '_editado'",
                font=("Arial", 8), fg="#e74c3c").pack(pady=10)
    
    def bring_to_front(self):
        self.root.attributes('-topmost', True)
        self.root.update()
        self.root.attributes('-topmost', False)
    
    def select_path(self, is_folder):
        self.bring_to_front()
        if is_folder:
            path = filedialog.askdirectory(title="Selecione a pasta com imagens originais")
        else:
            path = filedialog.askopenfilename(
                title="Selecione uma imagem para cortar",
                filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.JPG *.JPEG *.PNG *.BMP")]
            )
        self.bring_to_front()
        
        if path:
            threading.Thread(target=lambda: self.process(path, is_folder), daemon=True).start()
    
    def process(self, path, is_folder):
        self.bring_to_front()
        progress_win = tk.Toplevel(self.root)
        progress_win.title("Processando...")
        progress_win.geometry("400x100")
        progress_win.attributes('-topmost', True)
        tk.Label(progress_win, text="Aguarde, processando imagens...", font=("Arial", 11)).pack(pady=20)
        progress_win.update()
        
        try:
            if is_folder:
                images = [os.path.join(path, f) for f in os.listdir(path)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) 
                         and '_editado' not in f.lower()]
                total = len(images)
                if total == 0:
                    raise ValueError("Nenhuma imagem encontrada na pasta!")
                
                output_dir = os.path.join(path, "output")
                os.makedirs(output_dir, exist_ok=True)
                
                success = 0
                errors = []
                for i, img_path in enumerate(images, 1):
                    try:
                        cropped, coords = predict_crop(self.model, img_path, self.img_size)
                        base = os.path.splitext(os.path.basename(img_path))[0]
                        # Remover sufixos duplicados se existirem
                        base = base.replace('_editado', '').replace('_edited', '')
                        out_path = os.path.join(output_dir, f"{base}_editado.jpg")
                        cv2.imwrite(out_path, cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
                        success += 1
                        print(f"[{i}/{total}] ✅ {os.path.basename(img_path)} -> margens: x={coords[0]} y={coords[1]} w={coords[2]} h={coords[3]}")
                    except Exception as e:
                        errors.append(f"{os.path.basename(img_path)}: {str(e)}")
                        print(f"[{i}/{total}] ❌ {os.path.basename(img_path)}: {str(e)}")
                
                progress_win.destroy()
                self.bring_to_front()
                
                msg = f"Processamento concluído!\n\n" \
                      f"✓ Sucesso: {success}/{total}\n" \
                      f"✗ Erros: {len(errors)}\n\n" \
                      f"Saída: {output_dir}"
                if errors:
                    msg += f"\n\nPrimeiros erros:\n" + "\n".join(errors[:3])
                
                messagebox.showinfo("✅ Concluído", msg)
            else:
                cropped, coords = predict_crop(self.model, path, self.img_size)
                output_dir = os.path.join(os.path.dirname(path), "output")
                os.makedirs(output_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(path))[0]
                base = base.replace('_editado', '').replace('_edited', '')
                out_path = os.path.join(output_dir, f"{base}_editado.jpg")
                cv2.imwrite(out_path, cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
                
                progress_win.destroy()
                self.bring_to_front()
                
                messagebox.showinfo("✅ Sucesso", 
                                  f"Imagem cortada com precisão EXTREMA de margens!\n\n"
                                  f"Coordenadas: x={coords[0]} y={coords[1]} w={coords[2]} h={coords[3]}\n"
                                  f"Salva em: {out_path}")
        
        except Exception as e:
            progress_win.destroy()
            self.bring_to_front()
            messagebox.showerror("❌ Erro", f"Erro durante processamento:\n{str(e)}")
        finally:
            self.bring_to_front()


if __name__ == "__main__":
    root = tk.Tk()
    app = CropApp(root)
    root.mainloop()