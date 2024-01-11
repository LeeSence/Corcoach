from rest_framework import serializers

# Ocrlist의 queryset과 rest framework 연결
# 연결작업은 class로 처리한다.
#class Coach_picSerializer(serializers.ModelSerializer):
#    class Meta:
#        model = coach_pic
#        fields = ('id', 'username', 'dates', 'img')