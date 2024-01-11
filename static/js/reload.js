$(document).ready(function () {
    $('input[type=radio][name="SwitchCheck"]').change(function () {
      console.log($(this).val());
      value = $(this).val();
      console.log(value);
      $("#video").attr("src", value);
      $("#col-lg-6").load(window.location.href + "#col-lg-6");
    });
  });