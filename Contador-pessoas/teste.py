import cv2
import numpy as np

# Caminhos dos arquivos
ARQUIVO_VIDEO = 'Contador-pessoas/teste1.mp4'
ARQUIVO_MODELO = 'Contador-pessoas/frozen_inference_graph.pb'
ARQUIVO_CFG = 'Contador-pessoas/ssd_mobilenet_v2_coco.pbtxt'

def carregar_modelo(ARQUIVO_MODELO, ARQUIVO_CFG):
    '''
    Carrega o modelo de deep learning do TensorFlow para detecção de objetos.
    ARQUIVO_MODELO: Caminho para o arquivo .pb contendo os pesos do modelo.
    ARQUIVO_CFG: Caminho para o arquivo .pbtxt contendo a configuração do modelo.
    Retorna o modelo carregado.
    '''
    try:
        modelo = cv2.dnn.readNetFromTensorflow(ARQUIVO_MODELO, ARQUIVO_CFG)
    except cv2.error as erro:
        print(f"Erro ao carregar o modelo: {erro}")
        exit()
    return modelo

def aplicar_supressao_nao_maxima(caixas, confiancas, limiar_conf, limiar_supr):
    '''
    Aplica a Supressão Não Máxima para reduzir o número de caixas delimitadoras sobrepostas.
    caixas: Lista de caixas delimitadoras.
    confiancas: Lista de confianças de cada caixa.
    limiar_conf: Limiar de confiança para considerar detecções.
    limiar_supr: Limiar de sobreposição para suprimir caixas redundantes.
    Retorna uma lista de caixas após aplicar a supressão.
    '''
    indices = cv2.dnn.NMSBoxes(caixas, confiancas, limiar_conf, limiar_supr)
    return [caixas[i] for i in indices.flatten()] if len(indices) > 0 else []

def main():
    '''
    Função principal que executa o rastreio de pessoas no vídeo.
    '''
    captura = cv2.VideoCapture(ARQUIVO_VIDEO)
    detector_pessoas = carregar_modelo(ARQUIVO_MODELO, ARQUIVO_CFG)
    pausado = False
    contador_pessoas = 0
    linha_contagem = 400  # Posição x da linha de contagem
    detectados = []

    while True:
        if not pausado:
            ret, frame = captura.read()
            if not ret:
                break

            (altura, largura) = frame.shape[:2]
            if linha_contagem >= altura:
                linha_contagem = altura // 2  # Ajustar a linha para ficar no meio se estiver fora dos limites

            # Criação do blob a partir do frame e realização da detecção
            blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
            detector_pessoas.setInput(blob)
            deteccoes = detector_pessoas.forward()

            caixas = []
            confiancas = []

            # Extração das caixas delimitadoras e confianças das detecções
            for i in range(deteccoes.shape[2]):
                confianca = deteccoes[0, 0, i, 2]
                if confianca > 0.5:
                    caixa = deteccoes[0, 0, i, 3:7] * np.array([largura, altura, largura, altura])
                    (inicioX, inicioY, fimX, fimY) = caixa.astype("int")
                    caixas.append([inicioX, inicioY, fimX - inicioX, fimY - inicioY])
                    confiancas.append(float(confianca))

            # Aplicação da supressão não máxima para finalizar as caixas delimitadoras
            caixas_finais = aplicar_supressao_nao_maxima(caixas, confiancas, limiar_conf=0.5, limiar_supr=0.4)
            numero_pessoas = len(caixas_finais)

            # Verifica se pessoas cruzaram a linha de contagem
            novos_detectados = []
            for (inicioX, inicioY, largura, altura) in caixas_finais:
                centro_y = altura + inicioY // 2
                if linha_contagem - 10 < centro_y < linha_contagem + 10:
                    if inicioX not in detectados:
                        contador_pessoas += 1
                        detectados.append(inicioX)
                novos_detectados.append(inicioX)
                cv2.rectangle(frame, (inicioX, inicioY), (inicioX + largura, inicioY + altura), (0, 255, 0), 2)
            
            detectados = novos_detectados

            # Desenho da linha de contagem
            cv2.line(frame, (linha_contagem, 800), (linha_contagem, 65), (0, 0, 255), 2)

            # Exibição do número de pessoas detectadas e contadas
            cv2.putText(frame, f"Pessoas: {numero_pessoas}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Contador: {contador_pessoas}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Exibição do frame processado e controle de pausa/play
        cv2.imshow("Rastreio de Pessoas", frame)
        
        tecla = cv2.waitKey(30) & 0xFF
        if tecla == ord('q'):
            break
        elif tecla == ord('p'):
            pausado = not pausado

    # Liberação dos recursos ao finalizar
    captura.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
