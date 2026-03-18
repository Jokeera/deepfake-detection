# Final VKR Experiment Summary

- Total experiments: 4
- Successful: 3
- Failed: 1

## Best experiment

- Name: A3_temporal_only
- Model: temporal
- Fusion: adaptive
- Test AUC: 0.8177
- Test Accuracy: 0.7273
- Test F1: 0.7559

## Failed experiments

- A1_full_model: Размерности C,H,W не совпадают: spatial (3, 224, 224) vs temporal (3, 128, 128)

## Results table

```
======================================================================================================================
СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ
======================================================================================================================
Эксперимент               Status         Test AUC   Test Acc    Test F1    Epoch   Overfit?
-------------------------------------------------------------------------------------------
A1_full_model             failed           FAILED          —          —        —          —
A2_spatial_only           success        0.7760     0.7010     0.6864        1         OK
A3_temporal_only          success        0.8177     0.7273     0.7559        1         OK
A4_sequential             success        0.7767     0.7172     0.6759        1         OK
======================================================================================================================
```
